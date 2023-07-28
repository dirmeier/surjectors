"""
surjectors: Surjection layers for density estimation with normalizing flows
"""

__version__ = "0.2.3"

from surjectors.bijectors.lu_linear import LULinear
from surjectors.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.bijectors.permutation import Permutation
from surjectors.distributions.transformed_distribution import (
    TransformedDistribution,
)
from surjectors.surjectors.affine_masked_autoregressive_inference_funnel import (  # noqa: E501
    AffineMaskedAutoregressiveInferenceFunnel,
)
from surjectors.surjectors.affine_masked_coupling_generative_funnel import (
    AffineMaskedCouplingGenerativeFunnel,
)
from surjectors.surjectors.affine_masked_coupling_inference_funnel import (
    AffineMaskedCouplingInferenceFunnel,
)
from surjectors.surjectors.augment import Augment
from surjectors.surjectors.chain import Chain
from surjectors.surjectors.mlp import MLPFunnel
from surjectors.surjectors.rq_masked_coupling_inference_funnel import (
    RationalQuadraticSplineMaskedCouplingInferenceFunnel,
)
from surjectors.surjectors.slice import Slice
