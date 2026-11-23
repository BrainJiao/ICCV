"""OrthoID: Attribute-Nullspace Decoupling and Conflict-Free Cross-Paradigm Optimization."""

from .align import DifferentiableArcFaceAligner
from .budget import MidBandFilter, PublishableBudgetProjector, di_transform
from .irse50 import IRSE50Encoder
from .mask import FaceMaskBuilder
from .models import best_vae_dtype
from .optim import mgda_two_task
from .pfp import PFPConfig, PurificationFixedPointProtector, TweediePurifierOperator
from .protector import OrthoIDConfig, OrthoIDProtector
from .subspace import (
    DEFAULT_ATTRIBUTE_PAIRS,
    AttributeSubspace,
    build_photometric_basis,
    project_nullspace,
    pythagorean_counterfactual_target,
)

__all__ = [
    "AttributeSubspace",
    "DEFAULT_ATTRIBUTE_PAIRS",
    "DifferentiableArcFaceAligner",
    "FaceMaskBuilder",
    "IRSE50Encoder",
    "MidBandFilter",
    "OrthoIDConfig",
    "OrthoIDProtector",
    "PFPConfig",
    "PublishableBudgetProjector",
    "PurificationFixedPointProtector",
    "TweediePurifierOperator",
    "best_vae_dtype",
    "build_photometric_basis",
    "di_transform",
    "mgda_two_task",
    "project_nullspace",
    "pythagorean_counterfactual_target",
]
