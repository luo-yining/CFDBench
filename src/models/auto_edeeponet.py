from itertools import product
from typing import Dict, Optional

import torch
from torch import nn, Tensor

from .base_model import AutoCfdModel
from .ffn import Ffn
from .act_fn import get_act_fn
from .loss import MseLoss


class AutoEDeepONet(AutoCfdModel):
    """
    EDeepONet for autoregressive generation. The two input functions are
    the previous field (flattened) and the case parameters.

    Branch net accepts the boundary and physics properties as inputs.
    Trunk net accepts the query location (t, x, y) as input.
    """

    def __init__(
        self,
        dim_branch1: int,
        dim_branch2: int,
        trunk_dim: int,  # (x, y)
        loss_fn: MseLoss,
        num_label_samples: int = 1000,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        width: int = 100,
        act_name: str = "relu",
        act_norm: bool = False,
        act_on_output: bool = False,
    ):
        super().__init__(loss_fn)
        self.dim_branch1 = dim_branch1
        self.dim_branch2 = dim_branch2
        self.trunk_dim = trunk_dim
        self.loss_fn = loss_fn
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output
        self.num_label_samples = num_label_samples

        self.branch1_dims = [dim_branch1] + [width] * branch_depth
        self.branch2_dims = [dim_branch2] + [width] * branch_depth
        self.trunk_dims = [trunk_dim] + [width] * trunk_depth
        print(self.trunk_dims)

        act_fn = get_act_fn(act_name, act_norm)
        self.branch1 = Ffn(
            self.branch1_dims, act_fn=act_fn, act_on_output=self.act_on_output
        )
        self.branch2 = Ffn(
            self.branch2_dims, act_fn=act_fn, act_on_output=self.act_on_output
        )
        # we will be using an entire frame as label.
        self.trunk_net = Ffn(self.trunk_dims, act_fn=act_fn)

        self.bias = nn.Parameter(torch.zeros(1))  # type: ignore

    def forward(
        self,
        inputs: Tensor,  # (b, d1), previous field u(t-1)
        case_params: Tensor,  # (b, d2), physical properties
        label: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        query_idxs: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        A faster forward by using all the points in the frame (`label`) at
        time step `t` as training examples.

        Args:
        - x_branch: (b, branch_dim), input to the branch net.
        - t: (b), input to the trunk net, a batch of t
        - label: (b, w, h), the frame to be predicted.
        - query_idxs: (b, k, 2), the query locations.
        """

        # Add mask to input as additional channels
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, h, w)
            inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        batch_size, num_chan, height, width = inputs.shape
        inputs = inputs[:, 0]  # (B, h, w)
        flat_inputs = inputs.view(batch_size, -1)  # (B, h * w)

        b1 = self.branch1(flat_inputs)  # (b, p)
        b2 = self.branch2(case_params)  # (b, p)
        x_branch = b1 * b2

        if query_idxs is None:
            query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=flat_inputs.device,
            )  # (h * w, 2)

        # Normalize query location
        x_trunk = (query_idxs.float() - 50) / 100  # (k, 2)
        x_trunk = self.trunk_net(x_trunk)  # (k, p)
        x_trunk = x_trunk.unsqueeze(0)  # (1, k, p)
        x_branch = x_branch.unsqueeze(1)  # (b, 1, p)
        preds = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (b, k)

        # Use values of input field at query points as residuals
        residuals = inputs[:, query_idxs[:, 0], query_idxs[:, 1]]  # (b, k)
        preds = preds + residuals

        if label is not None:
            # Use only the u channel
            label = label[:, 0]  # (B, w, h)
            # we have labels[i, j] = label[
            #     i, query_points[i, j, 0], query_points[i, j, 1]]
            labels = label[:, query_idxs[:, 0], query_idxs[:, 1]]  # (b, k)
            assert preds.shape == labels.shape, f"{preds.shape}, {labels.shape}"
            loss = self.loss_fn(preds=preds, labels=labels)  # (b, k)
            return dict(
                preds=preds,
                loss=loss,
            )
        preds.view(-1, 1, height, width)
        return dict(preds=preds)

    def generate(
        self,
        x: Tensor,
        case_params: Tensor,
    ) -> Tensor:
        """
        Generate one frame at time t.

        Args:
        - x: Tensor, (b, field dim)
        - case_params: Tensor, (b, case params dim)
        - t: Tensor, (b)
        - height: int
        - width: int

        Returns:
            (b, c, h, w)
        """
        batch_size, num_chan, height, width = x.shape
        # Create 2D lattice of query points to infer the frame.
        query_idxs = torch.tensor(
            list(product(range(height), range(width))),
            dtype=torch.long,
            device=x.device,
        )  # (h * w, 2)
        # (b, 1, h * w)
        preds = self.forward(inputs=x, case_params=case_params, query_idxs=query_idxs)[
            "preds"
        ]
        preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
        return preds
