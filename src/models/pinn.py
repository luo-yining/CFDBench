from itertools import product
from typing import Dict, Optional

import torch
from torch import nn, Tensor

from .base_model import CfdNN
from .ffn import Ffn
from .act_fn import NormAct


def get_act_fn(name: str, norm: bool = False) -> nn.Module:
    if name == "relu":
        fn = nn.ReLU()
    elif name == "tanh":
        fn = nn.Tanh()
    elif name == "gelu":
        fn = nn.GELU()
    elif name == "swish":
        fn = nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    if norm:
        fn = NormAct(fn)
    return fn


class Pinn(CfdNN):
    """
    Data-driven PINN.

    Accepts the case parameters + the query coordinates as input.
    """

    def __init__(
        self,
        input_dim: int,
        loss_fn: nn.Module,
        branch_depth: int = 4,
        trunk_depth: int = 3,  # (t, x, y), or (x, y) for auto-regressive model.
        width: int = 100,
        act_name: str = "relu",
        act_norm: bool = False,
        act_on_output: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.loss_fn = loss_fn
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output

        act_fn = get_act_fn(act_name, act_norm)
        self.ffn = Ffn(
            self.branch_dims, act_fn=act_fn, act_on_output=self.act_on_output
        )
        self.fc = nn.Linear(self.trunk_dim, 1)  # outputs u(x, y, t)

    def forward(
        self,
        x: Tensor,
        label: Optional[Tensor] = None,
    ) -> dict:
        """
        Args:
        - x: (b, input_dim): the dim of case params + the dim of query coords.
        - label: (b), the label for the query location.

        Return:
        - if label is None:
            - preds: (b, k), the prediction for the query location.
        - else:
            - preds: (b, k), the prediction for the query location.
            - loss: (b, k), the loss for the query location.
        """

        x = self.ffn(x)
        preds = self.fc(x)

        if label is not None:
            loss = self.loss_fn(preds=preds, labels=label)  # (b, k)
            return dict(
                preds=preds,
                loss=loss,
            )
        return dict(preds=preds)

    def generate_one(
        self, x: Tensor, t: Tensor, height: int, width: int
    ) -> Tensor:
        """
        Generate one frame at time t.

        Args:
        - x: Tensor, (b, # case params)
        - t: Tensor, (b)
        - height: int
        - width: int

        Returns:
            (b, c, h, w)
        """
        # batch_size, num_chan, height, width = x_branch.shape

        # Create 2D lattice of query points to infer the frame.
        query_idxs = torch.tensor(
            list(product(range(height), range(width))),
            # dtype=torch.long,
            device=x.device,
        )  # (h * w, 2)
        # TODO: prepend each query point to each example's x in the batch.
        
        # (b, 1, h * w)
        output = self.forward(x_branch, t=t, query_idxs=query_idxs)["preds"]
        output = output.view(-1, 1, height, width)  # (b, 1, h, w)
        return output
