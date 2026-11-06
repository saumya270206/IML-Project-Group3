"""
SmoothGrad (Smilkov et al., 2017) implemented from scratch in PyTorch.

    M_hat(x) = (1/n) * sum_{i=1..n} M(x + N(0, sigma^2)),   sigma = noise_level * (x_max - x_min)

M(x) is the vanilla gradient saliency: d (class score) / d x. With n=1 and noise_level=0 this
reduces to the vanilla gradient. Also contains `smooth_gradcam`, the same noise-averaging idea
applied to Grad-CAM (an extra, not part of the original paper).

Design notes
* `target` may be None (predicted class of the CLEAN image), an int, a list of ints or a (B,) tensor.
  When it is None the class is fixed once on the clean input, so all noisy copies explain the same class.
* The model is put in eval() mode for the call and its previous mode is restored afterwards.
* `seed` makes the noise reproducible without touching the global RNG state.
* Only the input gradient is computed (torch.autograd.grad); parameter .grad fields are never written.
"""
import contextlib

import torch

from utils import minmax_normalise

MODES = ("mean", "abs", "square")
REDUCTIONS = ("max", "sum")


# ----------------------------------------------------------------------------- helpers
@contextlib.contextmanager
def _eval_mode(model):
    """eval() inside the block (BatchNorm uses running stats -> samples are independent), then restore."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _resolve_target(model, x, target):
    """None / int / list / tensor -> (B,) long tensor on x.device. None = argmax of the clean input."""
    if target is None:
        with torch.no_grad():
            return model(x).argmax(1)
    target = torch.as_tensor(target, device=x.device).long().reshape(-1)
    if target.numel() == 1:
        target = target.expand(x.shape[0])
    if target.numel() != x.shape[0]:
        raise ValueError(f"target has {target.numel()} entries but the batch has {x.shape[0]} images")
    return target


def _make_generator(x, seed):
    return None if seed is None else torch.Generator(device=x.device).manual_seed(int(seed))


def _noise_like(x, sigma, gen):
    """Gaussian noise with per-image std `sigma` (B,1,1,1), optionally from a seeded generator."""
    return torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * sigma


def _noise_std(x, noise_level):
    """sigma = noise_level * (x_max - x_min), computed per image as in the paper."""
    spread = x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(dim=(1, 2, 3), keepdim=True)
    return noise_level * spread


def _check_args(n_samples, noise_level, mode=None, reduce=None):
    if int(n_samples) < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if noise_level < 0:
        raise ValueError(f"noise_level must be >= 0, got {noise_level}")
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
    if reduce is not None and reduce not in REDUCTIONS:
        raise ValueError(f"reduce must be one of {REDUCTIONS}, got '{reduce}'")


def _collapse(g, reduce):
    """(B,3,H,W) non-negative gradients -> (B,H,W) saliency; 'max' or 'sum' over colour channels."""
    return g.amax(1) if reduce == "max" else g.sum(1)


def _clip_and_normalise(m, clip_percentile):
    """Optional visual clipping used in the paper: cap each map at its p-th percentile, then min-max to [0,1]."""
    if clip_percentile is not None:
        if not 0 < clip_percentile <= 100:
            raise ValueError(f"clip_percentile must be in (0, 100], got {clip_percentile}")
        hi = torch.quantile(m.flatten(1), clip_percentile / 100.0, dim=1).view(-1, 1, 1)
        m = torch.minimum(m, hi)
    return minmax_normalise(m)


# ----------------------------------------------------------------------------- core
def input_gradient(model, x, target):
    """Gradient of the pre-softmax score of `target` wrt the input. Returns (B,3,H,W)."""
    x = x.detach().clone().requires_grad_(True)               # guarantees a graph even if params are frozen
    target = _resolve_target(model, x, target)
    with torch.enable_grad():
        score = model(x).gather(1, target[:, None]).sum()     # samples independent in eval mode
        grad, = torch.autograd.grad(score, x)
    return grad


def vanilla_gradient(model, x, target=None, reduce="max", clip_percentile=None):
    """Plain gradient saliency |d y^c / d x|, collapsed over channels, normalised to [0,1]."""
    _check_args(1, 0.0, reduce=reduce)
    with _eval_mode(model):
        target = _resolve_target(model, x, target)
        g = input_gradient(model, x, target)
    return _clip_and_normalise(_collapse(g.abs(), reduce), clip_percentile)


def smoothgrad(model, x, target=None, n_samples=25, noise_level=0.15, mode="mean", reduce="max", seed=None,
               clip_percentile=None):
    """
    mode  : 'mean'   - average the raw gradients, take |.| afterwards (as in the paper)
            'abs'    - average |gradients|
            'square' - average squared gradients
    reduce: how to collapse colour channels, 'max' or 'sum'.
    seed  : int for reproducible noise (None = use the global RNG).
    clip_percentile: e.g. 99 caps each map at its 99th percentile before normalising (the paper does this
                     for display; ranking-based metrics are unaffected). None = no clipping.
    Returns a (B,H,W) map normalised to [0,1].
    """
    _check_args(n_samples, noise_level, mode, reduce)
    x = x.detach()
    gen = _make_generator(x, seed)
    with _eval_mode(model):
        target = _resolve_target(model, x, target)            # fixed on the clean image
        sigma = _noise_std(x, noise_level)                    # per-image noise std
        acc = torch.zeros_like(x)
        for _ in range(n_samples):
            g = input_gradient(model, x + _noise_like(x, sigma, gen), target)
            acc += g if mode == "mean" else (g.abs() if mode == "abs" else g * g)
    acc = (acc / n_samples).abs()
    return _clip_and_normalise(_collapse(acc, reduce), clip_percentile)


def smooth_gradcam(gradcam, x, target=None, n_samples=25, noise_level=0.15, seed=None):
    """Grad-CAM averaged over n noisy copies of the input. `gradcam` is a gradcam.GradCAM object."""
    _check_args(n_samples, noise_level)
    x = x.detach()
    gen = _make_generator(x, seed)
    with _eval_mode(gradcam.model):
        target = _resolve_target(gradcam.model, x, target)    # fixed on the clean image
    sigma = _noise_std(x, noise_level)
    cam = sum(gradcam(x + _noise_like(x, sigma, gen), target) for _ in range(n_samples)) / n_samples
    return minmax_normalise(cam)