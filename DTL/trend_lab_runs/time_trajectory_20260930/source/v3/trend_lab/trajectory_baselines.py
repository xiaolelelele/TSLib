"""Causal, bounded-budget adaptations of OCE-TS and CaReTS.

OCE-TS: Wang et al., AAAI 2026, Eqs. 2, 8--15, 18;
https://doi.org/10.1609/aaai.v40i31.39831
Official source commit a574b99e2e2673f7647b06ccda09de7d4b69f0e5.
Its test-time target-window extrema are deliberately replaced by fixed
training-only per-channel supports. This is not a full-paper replication.

CaReTS: https://arxiv.org/html/2511.09789v1, Eqs. 13--19, App. A.2/A.4.
The existing paper-formula branches are reused with an order-preserving
Transformer encoder; flatten/projection readout is an explicit adaptation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import _SeriesDecomposition
from .recent_methods import CaReTS


def ordinal_cross_entropy(predicted, target):
    """Eq. 2: SUM binary CE over the K-1 cumulative thresholds.

    The official code also sums thresholds. Float64 cumulative computation
    avoids a float32 1-epsilon clamp rounding to exactly one. No MSE or
    one-hot ordinal substitute is added.
    """
    if predicted.shape != target.shape or predicted.shape[-1] < 2:
        raise ValueError("probabilities must share shape and have at least two bins")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("ordinal probabilities must be finite")
    if torch.any(predicted < 0) or torch.any(target < 0):
        raise ValueError("ordinal probabilities must be nonnegative")
    for probabilities in (predicted, target):
        if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=2e-6):
            raise ValueError("ordinal probabilities must sum to one")
    cumulative = predicted.double().cumsum(-1)[..., :-1].clamp(1e-12, 1-1e-12)
    truth = target.double().cumsum(-1)[..., :-1]
    return -(truth * cumulative.log() + (1-truth) * torch.log1p(-cumulative)).sum(-1).mean()


class OCETimeSeries(nn.Module):
    """Official DLinear bin classifier with fixed TRAINING-only supports.

    ``train_min`` and ``train_max`` are [C] extrema in the runner's fixed
    standardized coordinates, computed solely from its training interval.
    Each channel is mapped affinely to [-1,1] for inputs AND target masses;
    probabilities are converted back with this SAME fixed affine map.
    Default K=45, sigma=.015 reproduce official run_longExp.py defaults.
    Targets outside support are explicitly boundary-clamped for soft-label
    construction; this affects training loss only, never metric labels.
    Outputs are inherently restricted to fixed bin-center support, a material
    limitation under drift. Inputs outside training extrema are not clipped.
    """
    def __init__(self, lookback, horizon, channels, train_min, train_max,
                 bins=45, sigma=.015, moving_average=25):
        super().__init__()
        if min(lookback, horizon, channels) < 1 or bins < 2:
            raise ValueError("positive dimensions and at least two bins required")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        lower = torch.as_tensor(train_min, dtype=torch.float64).detach().clone()
        upper = torch.as_tensor(train_max, dtype=torch.float64).detach().clone()
        if lower.shape != (channels,) or upper.shape != (channels,):
            raise ValueError("training extrema must have shape [channels]")
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or torch.any(upper < lower):
            raise ValueError("invalid training extrema")
        # A constant training variable receives a declared unit-width support;
        # division by zero and target-dependent range estimates are forbidden.
        constant = upper == lower
        lower = torch.where(constant, lower-.5, lower)
        upper = torch.where(constant, upper+.5, upper)
        self.lookback, self.horizon, self.channels, self.bins = lookback, horizon, channels, bins
        self.sigma = float(sigma)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        support = torch.linspace(-1., 1., bins+1, dtype=torch.float64)
        self.register_buffer("support", support)
        self.register_buffer("centers", (support[:-1]+support[1:])/2)
        self.decomposition = _SeriesDecomposition(moving_average)
        # Match the official shared-channel classifier, not the paper's more
        # compressed scalar->K sketch: two separate L -> H*K projections.
        self.seasonal = nn.Linear(lookback, horizon*bins)
        self.trend = nn.Linear(lookback, horizon*bins)

    def _check_x(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.lookback, self.channels):
            raise ValueError("wrong historical input dimensions")

    def _to_unit(self, values):
        return 2*(values-self.lower.to(values))/(self.upper-self.lower).to(values)-1

    def logits(self, x):
        self._check_x(x)
        seasonal, trend = self.decomposition(self._to_unit(x))
        result = self.seasonal(seasonal.transpose(1, 2)) + self.trend(trend.transpose(1, 2))
        return result.reshape(len(x), self.channels, self.horizon, self.bins).permute(0, 2, 1, 3)

    def probabilities(self, x):
        return self.logits(x).softmax(-1)

    def reconstruct(self, probabilities):
        if probabilities.shape[-2:] != (self.channels, self.bins):
            raise ValueError("probabilities must end in [channels,bins]")
        value = (probabilities*self.centers.to(probabilities)).sum(-1)
        return (value+1)*(self.upper-self.lower).to(value)/2+self.lower.to(value)

    def forward(self, x):
        return self.reconstruct(self.probabilities(x))

    def target_probabilities(self, y):
        if y.ndim != 3 or y.shape[1:] != (self.horizon, self.channels) or not torch.isfinite(y).all():
            raise ValueError("targets must be finite [batch,horizon,channels]")
        # Double precision CDF differences preserve Gaussian mass near the
        # boundaries; clamped centers are never used for evaluation labels.
        unit = self._to_unit(y.double()).clamp(-1., 1.)
        cdf = torch.erf((self.support-unit.unsqueeze(-1))/(self.sigma*math.sqrt(2)))
        masses = (cdf[..., 1:]-cdf[..., :-1]).clamp_min(0)
        return masses/masses.sum(-1, keepdim=True).clamp_min(1e-300)

    def loss(self, x, y):
        return ordinal_cross_entropy(self.probabilities(x), self.target_probabilities(y))


class _PhasePreservingEncoder(nn.Module):
    def __init__(self, lookback, channels, d_model, layers, heads, dropout):
        super().__init__()
        self.lookback, self.channels = lookback, channels
        self.input_projection = nn.Linear(channels, d_model)
        position = torch.arange(lookback, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)*(-math.log(10000.)/d_model))
        encoding = torch.zeros(lookback, d_model)
        encoding[:, 0::2] = torch.sin(position*frequency)
        encoding[:, 1::2] = torch.cos(position*frequency[:encoding[:, 1::2].shape[1]])
        self.register_buffer("position_encoding", encoding)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2*d_model,
                                            dropout=dropout, activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.readout = nn.Sequential(nn.Linear(lookback*d_model, 64), nn.ReLU())

    def forward(self, channel_first):
        if channel_first.shape[1:] != (self.channels, self.lookback):
            raise ValueError("wrong encoder input dimensions")
        temporal = self.input_projection(channel_first.transpose(1, 2))
        temporal = self.transformer(temporal+self.position_encoding[None, :, :].to(temporal))
        # Distinct learned coefficients for every time position retain phase
        # information; return [B,64,1] to reuse CaReTS's existing branches.
        return self.readout(temporal.flatten(1)).unsqueeze(-1)


class CaReTSPhaseTransformer(CaReTS):
    """CaReTS3/4 formulas, temporal-position-preserving Transformer encoder.

    Labels remain binary UP/DOWN RELATIVE TO LAST HISTORY (ties are UP),
    not adjacent raw three-class directions. Point MSE determines checkpoint
    selection externally. Input normalization, folds, budget and the flatten
    readout differ from the original paper; report a formula adaptation.
    """
    def __init__(self, lookback, horizon, channels, variant, d_model=64,
                 layers=2, heads=4, dropout=.1):
        if min(lookback, horizon, channels, d_model, layers, heads) < 1 or d_model % heads:
            raise ValueError("positive dimensions and d_model divisible by heads required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        super().__init__(lookback, horizon, channels, variant)
        self.encoder = _PhasePreservingEncoder(lookback, channels, d_model, layers, heads, dropout)

    def forward_with_true_anchor(self, x_float64):
        """Inference retaining the actual observed anchor and small deviations.

        The learned encoder/classification/regression branches still operate
        in the weights' dtype (normally float32). Their signed deviation is
        converted to float64 BEFORE adding the genuine last history value.
        Subtracting x32 from an already rounded forward(x32) cannot recover a
        swallowed deviation. Ordinary forward/loss training is unchanged.
        """
        if x_float64.dtype != torch.float64 or x_float64.ndim != 3 or x_float64.shape[1:] != (self.encoder.lookback,self.channels):
            raise ValueError("true-anchor inference requires original float64 [N,L,C] history")
        if not torch.isfinite(x_float64).all():
            raise ValueError("true-anchor history must be finite")
        weight=next(self.encoder.parameters())
        x=x_float64.to(device=weight.device,dtype=weight.dtype)
        h=self.encoder(x.transpose(1,2)).squeeze(-1)
        logits=self.classifier(h).reshape(len(x),self.horizon,self.channels,2)
        probability=logits.softmax(-1)
        if self.variant==3:
            deviation=F.relu(self.regressor(h)).reshape_as(probability)
            signed=probability[...,0]*deviation[...,0]-probability[...,1]*deviation[...,1]
        else:
            joined=torch.cat((h,probability.flatten(1)),dim=1)
            signed=self.regressor(joined).reshape(len(x),self.horizon,self.channels)
        anchor=x_float64[:,-1:,:].to(device=weight.device)
        return anchor+signed.to(dtype=torch.float64)
