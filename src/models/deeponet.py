from itertools import product
from typing import Dict, Optional

import torch
from torch import nn, Tensor

from .base_model import CfdModel
from .ffn import Ffn
from .act_fn import get_act_fn
from .loss import MseLoss


class DeepONet(CfdModel):
    """
    DeepONet for CFD.

    Branch net accepts the boundary and physics properties as inputs.
    Trunk net accepts the query location (t, x, y) as input.
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,  # (t, x, y)
        loss_fn: MseLoss,
        num_label_samples: int = 1000,
        branch_depth: int = 4,
        trunk_depth: int = 3,
        width: int = 100,
        act_name: str = "relu",
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
        self.loss_fn = loss_fn
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output
        self.num_label_samples = num_label_samples

        self.branch_dims = [branch_dim] + [width] * branch_depth

        act_fn = get_act_fn(act_name, act_norm)
        self.branch_net = Ffn(
            self.branch_dims, act_fn=act_fn, act_on_output=self.act_on_output
        )
        # t and x, y uses separate fc layers because during training
        # we will be using an entire frame as label.
        self.fc_trunk_t = nn.Linear(1, width)
        self.fc_trunk_xy = nn.Linear(2, width)
        self.trunk_dims = [width] * trunk_depth
        self.trunk_net = Ffn(self.trunk_dims, act_fn=act_fn)

        # self.params = self.__init_params()
        self.bias = nn.Parameter(torch.zeros(1))  # type: ignore
        # self.__initialize()

    def forward_vanilla(
        self,
        x_branch: Tensor,
        x_trunk: Tensor,
        label: Optional[Tensor] = None,
        query_idxs: Optional[Tensor] = None,
    ):
        """
        Args:
        - x_branch: (b, branch_dim), input to the branch net.
        - x_trunk: (b), input to the trunk net, a batch of (t, x, y)
        - label: (b), the label for the query location.

        Return:
        - if label is None:
            - preds: (b, k), the prediction for the query location.
        - else:
            - preds: (b, k), the prediction for the query location.
            - loss: (b, k), the loss for the query location.
        """

        # Create k query locations.
        # (b, k, 3), where each element is (t, x, y)
        if query_idxs is None:
            assert label is not None
            height, width = label.shape[-2:]
            query_idxs = torch.stack(
                [
                    torch.randint(
                        0, height, (self.num_label_samples,), device=x_trunk.device
                    ),
                    torch.randint(
                        0, width, (self.num_label_samples,), device=x_trunk.device
                    ),
                ],
                dim=-1,
            )  # (k, 2)
        t = x_trunk.unsqueeze(1).float()  # (b, 1)
        x_trunk_t = self.fc_trunk_t(t)  # (b, p)
        # Normalize query location
        x_trunk_xy = (query_idxs.float() - 32.0) / 64.0  # (k, 2)  # TODO: update this
        x_trunk_xy = self.fc_trunk_xy(x_trunk_xy)  # (k, p)
        x_trunk_t = x_trunk_t.unsqueeze(1)  # (b, 1, p)
        x_trunk_xy = x_trunk_xy.unsqueeze(0)  # (1, k, p)
        x_trunk = x_trunk_t + x_trunk_xy  # (b, k, p)

        x_branch = self.branch_net(x_branch)  # (b, p)
        x_trunk = self.trunk_net(x_trunk)  # (b, k, p)

        # x_trunk = x_trunk.unsqueeze(0)  # (1, k, p)
        x_branch = x_branch.unsqueeze(1)  # (b, 1, p)
        preds = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (b)
        print(preds.dtype)

        if label is not None:
            print(label.dtype)
            # Use only the u channel
            label = label[:, 0]  # (B, w, h)
            # we have labels[i, j] = label[
            #     i, query_points[i, j, 0], query_points[i, j, 1]]
            labels = label[:, query_idxs[:, 0], query_idxs[:, 1]]  # (b, k)
            # assert preds.shape == label.shape, f"{preds.shape}, {label.shape}"
            loss = self.loss_fn(preds=preds, labels=labels)  # (b, k)
            print(labels.dtype)
            print(loss.dtype)
            return preds, loss
        return preds

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

        x_trunk_t = self.fc_trunk_t(t)  # (b, p)
        # Normalize query location
        x_trunk_xy = query_idxs.float()  # (k, 2)
        x_trunk_xy = self.fc_trunk_xy(x_trunk_xy)  # (k, p)
        x_trunk_t = x_trunk_t.unsqueeze(1)  # (b, 1, p)
        x_trunk_xy = x_trunk_xy.unsqueeze(0)  # (1, k, p)
        x_trunk = x_trunk_t + x_trunk_xy  # (b, k, p)

        case_params = self.branch_net(case_params)  # (b, p)
        x_trunk = self.trunk_net(x_trunk)  # (b, k, p)

        # x_trunk = x_trunk.unsqueeze(0)  # (1, k, p)
        case_params = case_params.unsqueeze(1)  # (b, 1, p)
        preds = torch.sum(case_params * x_trunk, dim=-1) + self.bias  # (b, k)

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
