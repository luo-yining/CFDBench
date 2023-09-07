from itertools import product
from typing import List, Optional

import torch
from torch import Tensor

from .ffn import Ffn
from .base_model import AutoCfdModel
from .act_fn import get_act_fn
from .loss import MseLoss


class AutoFfn(AutoCfdModel):
    """
    Equivalent to autoregressive data-driven PINN.
    """

    def __init__(
        self,
        input_field_dim: int,
        num_case_params: int,
        query_dim: int,
        loss_fn: MseLoss,
        num_label_samples: int = 1000,
        depth: int = 8,
        width: int = 100,
        act_norm: bool = False,
        act_name="relu",
    ):
        '''
        Args:
        - branch_dim: int, the dimension of the branch net input.
        - trunk_dim: int, the dimension of the trunk net input.
        '''
        super().__init__(loss_fn)
        self.input_field_dim = input_field_dim
        self.num_case_params = num_case_params
        self.query_dim = query_dim
        self.depth = depth
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.num_label_samples = num_label_samples

        self.in_dim = input_field_dim + num_case_params + query_dim
        act_fn = get_act_fn(act_name, act_norm)
        self.widths = [self.in_dim] + [width] * depth + [1]
        self.ffn = Ffn(
            self.widths,
            act_fn=act_fn,
            act_on_output=False,
        )

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

        batch_size, _num_chan, height, width = inputs.shape

        # Only use the u channel
        inputs = inputs[:, 0]  # (B, h, w)
        flat_inputs = inputs.view(batch_size, -1)  # (B, h * w)
        flat_inputs = torch.cat([flat_inputs, case_params], dim=1)  # (B, h * w + 2)

        if query_idxs is None:
            query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=flat_inputs.device,
            )  # (k=h * w, 2)

            n_queries = query_idxs.shape[0]

            # For each combination of (input, query_point), we have a sample.
            # Repeat tensors such that we get (b * k) samples
            # (b, k, h * w)
            flat_inputs = flat_inputs.repeat(n_queries, 1)  # (b * k, h * w)
            batch_query_idxs = query_idxs.repeat(batch_size, 1)  # (b * k, 2)

            # (b * k, h * w + 2)
            flat_inputs = torch.cat([flat_inputs, batch_query_idxs.float()], dim=1)

        preds = self.ffn(flat_inputs)  # (b * k, 1)
        preds = preds.view(batch_size, -1)  # (b, k)

        # Use values of the input field at query points as residuals
        residuals = inputs[:, query_idxs[:, 0], query_idxs[:, 1]]  # (b, k)
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
