from torch import nn, Tensor
from typing import Optional

from .loss import MseLoss


class CfdModel(nn.Module):
    """
    Base model for all data-driven NN for CFD, learns the mapping from
    conditions (physics properties, boundary conditions and geometry)
    to the solution at a later time.
    """
    def __init__(self, loss_fn: MseLoss):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        label: Optional[Tensor] = None,
        case_params: Optional[dict] = None,
    ) -> dict:
        raise NotImplementedError

    def generate_one(
        self,
        case_params,
        t: Tensor,
        height: Tensor,
        width: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Generate the frame at time step `time`, given the case parameters.
        `case_params`.
        """
        raise NotImplementedError


class AutoCfdModel(nn.Module):
    """
    A CFD model that generates the solution auto-regressively, one frame at a time.
    """

    def __init__(self, loss_fn: MseLoss):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self,
        inputs: Tensor,
        label: Optional[Tensor] = None,
        case_params: Optional[dict] = None,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        raise NotImplementedError

    def generate(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Tensor,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError

    def generate_many(
        self,
        x: Tensor,
        case_params: Tensor,
        mask: Tensor,
        steps: int,
        **kwargs,
    ):
        """
        Given a frame `inputs`, generate the next `steps` frames.
        """
        raise NotImplementedError
