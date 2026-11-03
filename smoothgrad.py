"""
SmoothGrad (Smilkov et al., 2017) implemented from scratch in PyTorch.

    M_hat(x) = (1/n) * sum_{i=1..n} M(x + N(0, sigma^2)),   sigma = noise_level * (x_max - x_min)

M(x) is the vanilla gradient saliency: d (class score) / d x. With n=1 and noise_level=0 this
reduces to the vanilla gradient. Also contains `smooth_gradcam`, the same noise-averaging idea
applied to Grad-CAM (an extra, not part of the original paper).
"""
import torch

from utils import minmax_normalise


def input_gradient(model, x, target):
    """Gradient of the pre-softmax score of `target` wrt the input. Returns (B,3,H,W)."""
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        score = model(x).gather(1, target[:, None]).sum()   # samples independent in eval mode
        grad, = torch.autograd.grad(score, x)
    return grad


def _collapse(g, reduce):
    """(B,3,H,W) non-negative gradients -> (B,H,W) saliency; 'max' or 'sum' over colour channels."""
    return g.amax(1) if reduce == "max" else g.sum(1)


def vanilla_gradient(model, x, target, reduce="max"):
    """Plain gradient saliency |d y^c / d x|, collapsed over channels, normalised to [0,1]."""
    model.eval()
    return minmax_normalise(_collapse(input_gradient(model, x, target).abs(), reduce))


def smoothgrad(model, x, target, n_samples=25, noise_level=0.15, mode="mean", reduce="max"):
    """
    mode  : 'mean'   - average the raw gradients, take |.| afterwards (as in the paper)
            'abs'    - average |gradients|
            'square' - average squared gradients
    reduce: how to collapse colour channels, 'max' or 'sum'.
    Returns a (B,H,W) map normalised to [0,1].
    """
    model.eval()
    spread = x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(dim=(1, 2, 3), keepdim=True)
    sigma = noise_level * spread                              # per-image noise std
    acc = torch.zeros_like(x)
    for _ in range(n_samples):
        g = input_gradient(model, x + torch.randn_like(x) * sigma, target)
        acc += {"mean": g, "abs": g.abs(), "square": g ** 2}[mode]
    acc = (acc / n_samples).abs()
    return minmax_normalise(_collapse(acc, reduce))


def smooth_gradcam(gradcam, x, target, n_samples=25, noise_level=0.15):
    """Grad-CAM averaged over n noisy copies of the input. `gradcam` is a gradcam.GradCAM object."""
    spread = x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(dim=(1, 2, 3), keepdim=True)
    sigma = noise_level * spread
    cam = sum(gradcam(x + torch.randn_like(x) * sigma, target) for _ in range(n_samples)) / n_samples
    return minmax_normalise(cam)