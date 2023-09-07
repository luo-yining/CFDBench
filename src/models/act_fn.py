from torch import nn
from torch import Tensor


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


class NormAct(nn.Module):
    """
    Normalized Activation Function.

    A wrapper around any activation function that normalizes the input
    before applying the activation function, and then transforms the
    output back to the original scale.
    """
    def __init__(self, act_fn: nn.Module):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: (b, h, w)
        '''
        num_dims = len(x.shape)
        dims = tuple(range(1, num_dims))
        # find the mean and std of each example in the batch
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)
        # normalize
        x = (x - mean) / std
        x = self.act_fn(x)
        # Transform back to the original scale
        x = x * std + mean
        return x
