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
                in_chan,
                out_chan,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=bias,
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
        n_case_params: int,
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
        self.n_case_params = n_case_params
        self.hidden_chan = hidden_chan
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Create sequence of residual blocks
        blocks = [
            ResidualBlock(
                in_chan + 1 + n_case_params,  # + 1 for mask
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
        residual = inputs[:, : self.out_chan]
        batch_size, n_chan, height, width = inputs.shape
        if mask is None:
            mask = torch.ones((batch_size, height, width)).to(inputs.device)
        else:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, h, w)
        inputs = torch.cat([inputs, mask], dim=1)  # (B, c + 1, h, w)

        # Add case params as additional channels
        case_params = case_params.unsqueeze(-1).unsqueeze(-1)  # (B, c, 1, 1)
        # (B, n_params, h, w)
        case_params = case_params.expand(
            -1, -1, inputs.shape[-2], inputs.shape[-1]
        )
        inputs = torch.cat(
            [inputs, case_params], dim=1
        )  # (B, c + n_params, h, w)

        inputs = self.blocks(inputs)  # (B, c, h, w)
        preds = inputs + residual

        if mask is not None:
            # Mask out predictions.
            preds = preds * mask

        if label is not None:
            if mask is not None:
                # Mask out labels
                label = label * mask
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
        outputs = self.forward(inputs, case_params=case_params, mask=mask)
        preds = outputs["preds"]
        return preds

    def generate_many(
        self,
        inputs: Tensor,
        case_params: Tensor,
        steps: int,
        mask: Tensor,
    ):
        """
        x: (c, h, w)
        mask: (h, w). 1 for interior, 0 for boundaries.

        Returns:
            (steps, c, h, w)
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)  # (1, c, h, w)
            case_params = case_params.unsqueeze(0)  # (1, p)
            mask = mask.unsqueeze(0)  # (1, h, w)
        cur_frame = inputs  # (1, c, h, w)
        frames = [cur_frame]
        # boundaries = (1 - mask) * cur_frame  # (1, c, h, w)
        for _ in range(steps):
            cur_frame = self.generate(
                cur_frame, case_params=case_params, mask=mask
            )
            frames.append(cur_frame)
        return frames
