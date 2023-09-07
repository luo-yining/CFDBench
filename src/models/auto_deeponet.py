# coding: utf8

'''
Contains the implementation of autoregressive DeepONet
'''

from itertools import product
from typing import List, Optional

import torch
from torch import nn, Tensor

from .ffn import Ffn
from .base_model import AutoCfdModel
from .act_fn import get_act_fn
from .loss import MseLoss


class AutoDeepONet(AutoCfdModel):
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
        branch_dim: int,
        trunk_dim: int,
        loss_fn: MseLoss,
        num_label_samples: int = 1000,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        width: int = 100,
        act_name="relu",
        act_norm: bool = False,
        act_on_output: bool = False,
    ):
        '''
        Args:
        - branch_dim: int, the dimension of the branch net input.
        - trunk_dim: int, the dimension of the trunk net input.
        '''
        super().__init__(loss_fn)
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output
        self.num_label_samples = num_label_samples

        act_fn = get_act_fn(act_name, act_norm)
        self.branch_dims = [branch_dim] + [width] * branch_depth
        self.trunk_dims = [trunk_dim] + [width] * trunk_depth
        self.branch_net = Ffn(
            self.branch_dims,
            act_fn=act_fn,
            act_on_output=act_on_output,
        )
        self.trunk_net = Ffn(self.trunk_dims, act_fn=act_fn)
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

        Args:
            inputs: (b, c, h, w)
            case_params: (b, p)
            labels: (b, c, h, w)
            query_point: (k, 2), k is the number of query points, each is an (x, y)
                coordinate.
            masks: For future use.

        Returns:
            Output: Tensor, if query_points is not None, the shape is (b, k).
                Else, the shape is (b, c, h, w).

        Notations:
        - b: batch size
        - c: number of channels
        - h: height
        - w: width
        - p: number of case parameters
        - k: number of query points
        """

        # Add mask to input as additional channels
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, h, w)
            inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        batch_size, num_chan, height, width = inputs.shape

        # Only use the u channel
        inputs = inputs[:, 0]  # (B, h, w)
        flat_inputs = inputs.view(batch_size, -1)  # (B, h * w)

        # Simple prepend physical properties to the input field.
        flat_inputs = torch.cat([flat_inputs, case_params], dim=1)  # (B, h * w + 2)
        x_branch = self.branch_net(flat_inputs)

        if query_idxs is None:
            query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=flat_inputs.device,
            )  # (h * w, 2)

        # Input to the trunk net
        x_trunk = (query_idxs.float() - 50) / 100  # (k, 2)
        x_trunk = self.trunk_net(x_trunk)  # (k, p)
        x_trunk = x_trunk.unsqueeze(0)  # (1, k, p)
        x_branch = x_branch.unsqueeze(1)  # (b, 1, p)
        preds = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (b, k)

        # Use values of the input field at query points as residuals
        residuals = inputs[:, query_idxs[:, 0], query_idxs[:, 1]]  # (b, k)
        preds += residuals

        if label is not None:
            label = label[:, 0]  # (B, 1, h, w)  # Predict only u
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
        preds = self.forward(x, query_idxs=query_idxs, case_params=case_params)['preds']
        preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
        return preds

    def generate_many(
        self, x: Tensor, case_params: Tensor, steps: int
    ) -> List[Tensor]:
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
