from typing import List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MseLoss(nn.Module):
    def __init__(self, normalize: bool, is_masked: bool = False):
        super().__init__()
        self.normalize = normalize
        self.is_masked = is_masked

    def get_score_names(self) -> List[str]:
        names = [
            'mse', 'rmse', 'mae'
        ]
        if self.normalize:
            names += ['nmse']
        return names

    def forward(self, preds: Tensor, labels: Tensor) -> dict[str, Tensor]:
        '''
        Args:
        - mask: 1 for valid pixels, 0 for invalid pixels.
        '''
        mse = F.mse_loss(input=preds, target=labels)
        mae = F.l1_loss(input=preds, target=labels)
        result = dict(
            mse=mse,
            rmse=torch.sqrt(mse),
            mae=mae,
        )
        if self.normalize:
            result['nmse'] = mse / torch.square(labels).mean()
            # result['mre'] = mae / torch.abs(labels).mean()
        return result


def loss_name_to_fn(name: str, masked: bool = False) -> MseLoss:
    name = name.lower()
    if masked:
        raise NotImplementedError
    else:
        if name == 'mse':
            return MseLoss(normalize=False)
        elif name == 'nmse':
            return MseLoss(normalize=True)
        else:
            raise NotImplementedError
