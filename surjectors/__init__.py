"""
surjectors: Surjection layers for density estimation with normalizing flows
"""

__version__ = "0.1.2"

from surjectors.bijectors.lu_linear import LULinear
from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.conditioners import mlp_conditioner, transformer_conditioner
from surjectors.distributions.transformed_distribution import (
    TransformedDistribution,
)
from surjectors.surjectors.affine_masked_coupling_generative_funnel import (
    AffineMaskedCouplingGenerativeFunnel,
)
from surjectors.surjectors.affine_masked_coupling_inference_funnel import (
    AffineMaskedCouplingInferenceFunnel,
)
from surjectors.surjectors.augment import Augment
from surjectors.surjectors.chain import Chain
from surjectors.surjectors.mlp import MLP
from surjectors.surjectors.rq_coupling_funnel import NSFCouplingFunnel
from surjectors.surjectors.slice import Slice
