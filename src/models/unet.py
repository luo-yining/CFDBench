""" Parts of the U-Net model """
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base_model import AutoCfdModel


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self, in_chan: int, out_chan: int, mid_chan: Optional[int] = None
    ):
        super().__init__()
        if mid_chan is None:
            mid_chan = out_chan
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_chan,
                mid_chan,
                kernel_size=3,
                padding=1,
                bias=True,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(mid_chan),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan,
                out_chan,
                kernel_size=3,
                padding=1,
                bias=True,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number
        # of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(AutoCfdModel):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        loss_fn: nn.Module,
        n_case_params: int,
        insert_case_params_at: str = "hidden",
        bilinear: bool = False,
        dim: int = 8,
    ):
        assert insert_case_params_at in ["hidden", "input"]

        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.insert_case_params_at = insert_case_params_at
        self.bilinear = bilinear
        self.dim = dim

        if insert_case_params_at == "hidden":
            self.case_params_fc = nn.Linear(n_case_params, dim * 16)

        if insert_case_params_at == "input":
            self.in_conv = DoubleConv(
                in_chan + 1 + n_case_params, dim   # + 1 for mask
            )
        else:
            self.in_conv = DoubleConv(in_chan + 1, dim)  # + 1 for mask
        self.down1 = Down(dim, dim * 2)
        self.down2 = Down(dim * 2, dim * 4)
        self.down3 = Down(dim * 4, dim * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor)
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.out_conv = OutConv(dim, out_chan)

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ):
        """
        x: (B, c, h, w)
        mask: (B, h, w), mask out parts of the image that is an obstacle.
        label: (B, c, h, w)
        case_params; (b, n_params)
        """
        batch_size, n_chan, height, width = inputs.shape
        residual = inputs[:, : self.out_chan]

        # Add mask to input as additional channels
        if mask is None:
            mask = torch.ones((batch_size, height, width)).to(inputs.device)
        else:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, h, w)
        inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        # Add case_params to input as additional channels
        if self.insert_case_params_at == "input":
            assert case_params is not None
            case_params = case_params.unsqueeze(2).unsqueeze(
                3
            )  # (B, n_params, 1, 1)
            # (B, n_params, h, w)
            case_params = case_params.expand(
                -1, -1, inputs.shape[2], inputs.shape[3]
            )
            inputs = torch.cat(
                [inputs, case_params], dim=1
            )  # (B, c + 1, h, w)

        x1 = self.in_conv(inputs)  # (B, dim, h, w)
        x2 = self.down1(x1)  # (B, dim * 2, h/2, w/2)
        x3 = self.down2(x2)  # (B, dim * 4, h/4, w/4)
        x4 = self.down3(x3)  # (B, dim * 8, h/8, w/8)
        x5 = self.down4(x4)  # (B, dim * 16, h/16, w/16)

        # Add case params to the hidden representation
        if self.insert_case_params_at == "hidden":
            assert case_params is not None
            assert self.case_params_fc is not None
            conds = self.case_params_fc(case_params)  # (B, dim * 16)
            assert conds is not None
            conds = conds.unsqueeze(2).unsqueeze(3)  # (B, dim * 16, 1, 1)
            x5 = x5 + conds

        inputs = self.up1(x5, x4)  # (B, dim * 8, h/8, w/8)
        inputs = self.up2(inputs, x3)  # (B, dim * 4, h/4, w/4)
        inputs = self.up3(inputs, x2)  # (B, dim * 2, h/2, w/2)
        inputs = self.up4(inputs, x1)  # (B, dim, h, w)
        preds = self.out_conv(inputs)  # (B, out_chan, h, w)
        preds += residual

        preds = preds * mask

        if label is not None:
            label = label * mask

            loss: dict = self.loss_fn(labels=label, preds=preds)
            return dict(
                preds=preds,
                loss=loss,
            )
        return dict(preds=preds)

    def generate_many(
        self, inputs: Tensor, case_params: Tensor, mask: Tensor, steps: int
    ) -> List[Tensor]:
        """
        Generate multiple steps of the solution autoregressively.

        x: (c, h, w) or (B, c, h, w)
        mask: (h, w). 1 for interior, 0 for boundaries.

        Returns:
            (steps, c, h, w)
        """
        preds = []
        if inputs.dim() == 3:
            # Add dim for batch size
            inputs = inputs.unsqueeze(0)  # (1, c, h, w)
            case_params = case_params.unsqueeze(0)
            mask = mask.unsqueeze(0)
        cur_frame = inputs  # (b, c, h, w)
        # boundaries = (1 - mask) * inputs
        mask = mask.unsqueeze(0)
        for _ in range(steps):
            # (b, c, h, w)
            cur_frame = self.generate(
                cur_frame, case_params=case_params, mask=mask
            )
            preds.append(cur_frame)
        return preds

    def generate(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = self.forward(inputs, case_params=case_params, mask=mask)
        preds = outputs["preds"]
        return preds
