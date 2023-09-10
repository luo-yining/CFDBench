from typing import Optional

import torch
from torch import nn, Tensor


from .base_model import AutoCfdModel


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        hidden_chan: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dropout_rate: float = 0.2,
        bias: bool = True,
        use_1x1conv: bool = False,
    ):
        super().__init__()
        if in_chan != out_chan:
            assert use_1x1conv
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.hidden_chan = hidden_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Sub-modules
        self.conv1 = nn.Conv2d(
            in_chan,
            hidden_chan,
            kernel_size,
            stride,
            padding,
            bias=bias,
            padding_mode="replicate",
        )
        self.bn1 = nn.BatchNorm2d(hidden_chan)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            hidden_chan,
            out_chan,
            kernel_size,
            stride,
            padding,
            bias=bias,
            padding_mode="replicate",
        )
        self.bn2 = nn.BatchNorm2d(out_chan)

        if use_1x1conv:
            self.res_conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, padding=0, bias=bias
            )
        else:
            self.res_conv = None

    def forward(self, x: Tensor) -> Tensor:
        if self.res_conv is not None:
            residual = self.res_conv(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.conv2(x)
        x += residual
        return x


class ResNet(AutoCfdModel):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        loss_fn: nn.Module,
        hidden_chan: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 7,
        padding: int = 3,
        stride: int = 1,
    ):
        super().__init__(loss_fn)
        assert in_chan == out_chan
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.hidden_chan = hidden_chan
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Create sequence of residual blocks
        blocks = [
            ResidualBlock(
                in_chan + 1 + 5,  # +3 for case params
                hidden_chan,
                64,
                kernel_size,
                stride,
                padding,
                use_1x1conv=True,
            ),
        ]
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    hidden_chan,
                    hidden_chan,
                    64,
                    kernel_size,
                    stride,
                    padding,
                    use_1x1conv=False,
                )
            )
        blocks.append(
            ResidualBlock(
                hidden_chan,
                out_chan,
                64,
                kernel_size,
                stride,
                padding,
                use_1x1conv=True,
            )
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
    ) -> dict:
        """
        Args:
        - x: (B, in_chan, h, w)
        - case_params: (B, 3)
        - mask: (h, w) or (B, h, w). 1 for interior, 0 for boundaries.
        - label: (B, out_chan, h, w)

        Returns:
            (B, out_chan, h, w) or (B, out_chan, h, w), loss
        """
        residual = inputs[:, :self.out_chan]
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # mask = mask.unsqueeze(1)  # (B, 1, h, w)
            inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        # Add case params as additional channels
        case_params = case_params.unsqueeze(-1).unsqueeze(-1)  # (B, c, 1, 1)
        # (B, n_params, h, w)
        case_params = case_params.expand(-1, -1, inputs.shape[-2], inputs.shape[-1])
        inputs = torch.cat([inputs, case_params], dim=1)  # (B, c + n_params, h, w)

        inputs = self.blocks(inputs)  # (B, c, h, w)
        preds = inputs + residual
        if label is not None:
            loss = self.loss_fn(preds=preds, labels=label)
            return dict(
                preds=preds,
                loss=loss,
            )
        else:
            return dict(preds=preds)

    def generate(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
    ):
        """
        Args:
        - mask: 1 for interior
        """
        outputs = self.forward(inputs, case_params=case_params, mask=mask)
        preds = outputs["preds"]
        if mask is not None:
            return preds * mask + (1 - mask) * inputs[:, 0]
        return preds

    def generate_many(
        self, x: Tensor, case_params: Tensor, steps: int, mask: Optional[Tensor] = None
    ):
        """
        x: (c, h, w)
        mask: (h, w). 1 for interior, 0 for boundaries.

        Returns:
            (steps, c, h, w)
        """
        frames = []
        cur_frame = x[:2].unsqueeze(0)  # (1, c, h, w)
        boundaries = (1 - mask) * cur_frame  # (1, c, h, w)
        mask = mask.unsqueeze(0)  # (1, h, w)
        mask = mask.unsqueeze(1)  # (1, 1, h, w)
        for _ in range(steps):
            outputs = self.forward(cur_frame, case_params=case_params, mask=mask)
            cur_frame = outputs * mask + boundaries
            frames.append(cur_frame.squeeze(0))
        return frames
