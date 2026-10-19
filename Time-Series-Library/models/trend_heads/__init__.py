from .linear import TrendHeadLinearPerDim
from .mlp import TrendHeadMLPPerDim
from .conv import TrendHeadConv1x1
from .separate import TrendHeadPerDimMLPSeparate
from .context import TrendContextHead

trend_head_registry = {
    'linear': TrendHeadLinearPerDim,
    'mlp': TrendHeadMLPPerDim,
    'conv': TrendHeadConv1x1,
    'separate': TrendHeadPerDimMLPSeparate,
    'context': TrendContextHead
}
