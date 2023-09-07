from typing import Optional, Dict, List
from itertools import product

from torch import nn, Tensor
import torch

from .base_model import CfdModel
from .loss import MseLoss
from .act_fn import get_act_fn


class Ffn(nn.Module):
    """
    A fully connected multi-layer neural network.
    """

    def __init__(self, dims: list, act_fn: nn.Module, act_on_output: bool = False):
        super().__init__()
        self.dims = dims

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn)
            # layers.append(NormAct(nn.Tanh()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if act_on_output:
            layers.append(act_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class FfnModel(CfdModel):
    """
    DeepONet for CFD.

    Branch net accepts the boundary and physics properties as inputs.
    Trunk net accepts the query location (t, x, y) as input.
    """

    def __init__(
        self,
        loss_fn: MseLoss,
        widths: List[int],
        act_name: str = "relu",
        act_norm: bool = True,
        act_on_output: bool = False,
        num_label_samples: int = 1000,
    ):
        '''
        Args:
        - branch_dim: int, the dimension of the branch net input.
        - trunk_dim: int, the dimension of the trunk net input.
        '''
        super().__init__(loss_fn)
        self.loss_fn = loss_fn
        self.widths = widths
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output
        self.num_label_samples = num_label_samples

        act_fn = get_act_fn(act_name, act_norm)
        self.ffn = Ffn(
            self.widths, act_fn=act_fn, act_on_output=self.act_on_output
        )

    def forward(
        self,
        case_params: Tensor,  # Case parameters
        t: Tensor,
        label: Optional[Tensor] = None,
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

        batch_size, dim_in = case_params.shape

        if query_idxs is None:
            # Create k query locations on a lattice.
            # (b, k, 2), where each element is (x, y)
            assert label is not None
            height, width = label.shape[-2:]
            query_idxs = torch.stack(
                [
                    torch.randint(
                        0, height, (self.num_label_samples,), device=label.device
                    ),
                    torch.randint(
                        0, width, (self.num_label_samples,), device=label.device
                    ),
                ],
                dim=-1,
            )  # (k, 2)

        # Concatenate query spatial coordinates with time coordinate
        coords = query_idxs.unsqueeze(0)  # (1, k, 2)
        coords = coords.repeat(batch_size, 1, 1)  # (b, k, 2)
        num_queries = coords.shape[1]
        t = t.unsqueeze(-1)  # (b, 1)
        t = t.repeat(1, num_queries, 1)  # (b, k)
        coords = torch.cat([coords, t], dim=-1)  # (b, k, 3)

        # Concatenate case params with queries
        case_params = case_params.unsqueeze(1)  # (b, 1, p)
        case_params = case_params.repeat(1, num_queries, 1)  # (b, k, p)
        inp = torch.cat([case_params, coords], dim=-1)  # (b, k, p + 3)
        inp = inp.view(batch_size * num_queries, -1)  # (bk, p + 3)
        preds = self.ffn(inp)  # (bk, out_dim)
        preds = preds.view(batch_size, num_queries)  # (b, k)

        if label is not None:
            # Use only the u channel
            label = label[:, 0]  # (B, w, h)
            labels = label[:, query_idxs[:, 0], query_idxs[:, 1]]  # (b, k)
            assert preds.shape == labels.shape, f"{preds.shape}, {labels.shape}"
            loss = self.loss_fn(preds=preds, labels=labels)  # (b, k)
            return dict(
                preds=preds,
                loss=loss,
            )
        return dict(
            preds=preds,
        )

    def generate_one(
        self, case_params: Tensor, t: Tensor, height: int, width: int
    ) -> Tensor:
        """
        Generate one frame at time t.

        Args:
        - x_branch: Tensor, (b, branch_dim)
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
            device=case_params.device,
        )  # (h * w, 2)

        # query_points = query_points / 100
        # (b, 1, h * w)
        output = self.forward(case_params, t=t, query_idxs=query_idxs)["preds"]
        output = output.view(-1, 1, height, width)  # (b, 1, h, w)
        return output
