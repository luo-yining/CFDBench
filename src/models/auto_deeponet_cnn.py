from itertools import product
from typing import List, Optional

import torch
from torch import nn, Tensor

from .ffn import Ffn
from .base_model import AutoCfdModel
from .act_fn import get_act_fn
from .loss import MseLoss


class CnnBranch(nn.Module):
    def __init__(self, in_chan: int, kernel_size: int, padding: int, depth: int = 4):
        super().__init__()
        self.in_chan = in_chan
        self.in_conv = nn.Conv2d(in_chan, 32, kernel_size=kernel_size, padding=padding)
        self.out_conv = nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding)

        blocks = []
        for i in range(depth):
            blocks += [
                nn.Conv2d(32, 32, kernel_size, padding=padding),
                nn.MaxPool2d(2),
                nn.ReLU(),
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)  # (b, 16, h, w)
        x = self.blocks(x)  # (b, 32, h/16=4, w/16=4)
        x = self.out_conv(x)  # (b, 32, 4, 4)
        return x


class AutoDeepONetCnn(AutoCfdModel):
    """
    Auto-regressive DeepONet for CFD.

    Our task is different from the one that the original DeepONet. In the
    original DeepONet, the input function (input to the branch net)
    is the initial condition (IC), but here, we have a fixed (zero) IC.
    Instead, we have different boundary conditions (BCs), but we also
    want the model to predict the next time step given the current time step.
    Ideally, we should have two different branch nets, one accepting the
    BCs, one accepting the current time step.

    Here, we assume that the current time step includes the information about
    BCs (which are the values on the bounds), so we just feed
    the current time step to one branch net.
    """

    def __init__(
        self,
        in_chan: int,
        query_dim: int,
        loss_fn: MseLoss,
        height: int = 100,
        width: int = 100,
        num_case_params: int = 5,
        trunk_depth: int = 4,
        act_name="relu",
        act_norm: bool = False,
        act_on_output: bool = False,
    ):
        """
        Args:
        - branch_dim: int, the dimension of the branch net input.
        - trunk_dim: int, the dimension of the trunk net input.
        """
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.query_dim = query_dim
        self.num_case_params = num_case_params
        self.trunk_depth = trunk_depth
        self.height = height
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output

        act_fn = get_act_fn(act_name, act_norm)
        self.trunk_dims = [query_dim] + [100] * trunk_depth + [4 * 4 * 32]
        self.branch_net = CnnBranch(in_chan + num_case_params, kernel_size=5, padding=2)
        self.trunk_net = Ffn(self.trunk_dims, act_fn=act_fn, act_on_output=False)
        self.out_ffn = Ffn([32 * 4 * 4] * 3 + [1], act_fn=act_fn, act_on_output=False)
        self.bias = nn.Parameter(torch.zeros(1))  # type: ignore

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        label: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        query_idxs: Optional[Tensor] = None,
    ):
        """
        Here, we just randomly sample some points, and use the label values on
        those points as the label.

        x: (b, c, h, w)
        labels: (b, c, h, w)
        query_point: (k, 2), k is the number of query points, each is an (x, y)
            coordinate.

        Goal:
        Input: [b, branch_dim + trunk_dim]
        Output: [b, 1]
        """

        # Add mask to input as additional channels
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, h, w)
            inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        batch_size, num_chan, height, width = inputs.shape

        # Only use the u channel
        residuals = inputs[:, : self.in_chan]  # (B, c, h, w)

        # Add case params as additional channels
        case_params = case_params.unsqueeze(-1).unsqueeze(-1)  # (B, c, 1, 1)
        # (B, n_params, h, w)
        case_params = case_params.expand(-1, -1, inputs.shape[-2], inputs.shape[-1])
        inputs = torch.cat([inputs, case_params], dim=1)  # (B, c + n_params, h, w)

        x_branch = self.branch_net(inputs)  # (b, 32, h/16=4, w/16=4)
        x_branch = x_branch.view(batch_size, -1)  # (b, 32 * 4 * 4 = 512)

        if query_idxs is None:
            query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=x_branch.device,
            )  # (h * w, 2)

        # Input to the trunk net
        x_trunk = (query_idxs.float() - 50) / 100  # (k, 2)
        x_trunk = self.trunk_net(x_trunk)  # (k, p)
        x_trunk = x_trunk.unsqueeze(0)  # (1, k, p)
        x_branch = x_branch.unsqueeze(1)  # (b, 1, 32, p)
        # preds = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (b, k)
        preds = x_branch * x_trunk  # (b, k, h)
        preds = self.out_ffn(preds)  # (b, k, 1)
        preds = preds.squeeze(-1)  # (b, k)

        # Use values of the input field at query points as residuals
        residuals = residuals[:, 0, query_idxs[:, 0], query_idxs[:, 1]]  # (b, c, k)
        # print(residuals.shape, preds.shape)
        preds += residuals

        if label is not None:
            label = label[:, 0]  # (B, 1, h, w)  # Predict only u
            # we have labels[i, j] = label[
            #     i, query_points[i, j, 0], query_points[i, j, 1]]
            labels = label[:, query_idxs[:, 0], query_idxs[:, 1]]  # (b, k)
            loss = self.loss_fn(labels=labels, preds=preds)  # (b, k)
            return dict(
                preds=preds,
                loss=loss,
            )

        preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
        return dict(preds=preds)

    def generate(self, x: Tensor, case_params: Tensor) -> Tensor:
        """
        x: (c, h, w) or (B, c, h, w)

        Returns:
            (b, c, h, w)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, c, h, w)
        batch_size, num_chan, height, width = x.shape
        query_idxs = torch.tensor(
            list(product(range(height), range(width))),
            dtype=torch.long,
            device=x.device,
        )  # (h * w, 2)
        # query_points = query_points / 100
        # (b, 1, h * w)
        preds = self.forward(x, query_idxs=query_idxs, case_params=case_params)["preds"]
        preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
        return preds

    def generate_many(self, x: Tensor, case_params: Tensor, steps: int) -> List[Tensor]:
        """
        x: (c, h, w) or (B, c, h, w)
        mask: (h, w). 1 for interior, 0 for boundaries.
        steps: int, number of steps to generate.

        Returns:
            list of tensors, each of shape (b, c, h, w)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, c, h, w)
        cur_frame = x
        frames = [cur_frame]
        for _ in range(steps):
            # (b, c, h, w)
            cur_frame = self.generate(cur_frame, case_params=case_params)
            frames.append(cur_frame)
        return frames
