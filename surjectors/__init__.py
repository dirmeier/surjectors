"""
surjectors: Surjection layers for density estimation with normalizing flows
"""

__version__ = "0.3.0"

from surjectors._src.bijectors.lu_linear import LULinear
from surjectors._src.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors._src.bijectors.masked_coupling import MaskedCoupling
from surjectors._src.bijectors.permutation import Permutation
from surjectors._src.distributions.transformed_distribution import (
    TransformedDistribution,
)
from surjectors._src.surjectors.affine_masked_autoregressive_inference_funnel import (  # noqa: E501
    AffineMaskedAutoregressiveInferenceFunnel,
)
from surjectors._src.surjectors.affine_masked_coupling_generative_funnel import (  # noqa: E501
    AffineMaskedCouplingGenerativeFunnel,
)
from surjectors._src.surjectors.affine_masked_coupling_inference_funnel import (
    AffineMaskedCouplingInferenceFunnel,
)
from surjectors._src.surjectors.augment import Augment
from surjectors._src.surjectors.chain import Chain
from surjectors._src.surjectors.masked_autoregressive_inference_funnel import (  # noqa: E501
    MaskedAutoregressiveInferenceFunnel,
)
from surjectors._src.surjectors.masked_coupling_inference_funnel import (
    MaskedCouplingInferenceFunnel,
)
from surjectors._src.surjectors.mlp import MLPInferenceFunnel
from surjectors._src.surjectors.rq_masked_autoregressive_inference_funnel import (  # noqa: E501
    RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel,
)
from surjectors._src.surjectors.rq_masked_coupling_inference_funnel import (
    RationalQuadraticSplineMaskedCouplingInferenceFunnel,
)
from surjectors._src.surjectors.slice import Slice

__all__ = [
    "Chain",
    "Permutation",
    "TransformedDistribution",
    "MaskedAutoregressive",
    "MaskedAutoregressiveInferenceFunnel",
    "AffineMaskedAutoregressiveInferenceFunnel",
    "RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel",
    "MaskedCoupling",
    "MaskedCouplingInferenceFunnel",
    "AffineMaskedCouplingInferenceFunnel",
    "RationalQuadraticSplineMaskedCouplingInferenceFunnel",
    # "AffineMaskedCouplingGenerativeFunnel",
    "LULinear",
    "MLPInferenceFunnel",
    "Slice",
    # "Augment",
]
