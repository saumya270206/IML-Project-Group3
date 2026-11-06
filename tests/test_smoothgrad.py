import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torchvision

from gradcam import GradCAM
from smoothgrad import input_gradient, vanilla_gradient, smoothgrad, smooth_gradcam
from utils import minmax_normalise


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None).eval()   # random weights: no download needed


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(2, 3, 64, 64)                           # small images keep the tests fast


# ----------------------------------------------------------------------------- basic contract
def test_output_shape_and_range(model, x):
    for mode in ("mean", "abs", "square"):
        for reduce in ("max", "sum"):
            m = smoothgrad(model, x, None, n_samples=3, mode=mode, reduce=reduce, seed=0)
            assert m.shape == (2, 64, 64)
            assert m.min() >= 0 and m.max() <= 1
            assert not torch.isnan(m).any()


def test_each_map_reaches_zero_and_one(model, x):
    m = smoothgrad(model, x, None, n_samples=3, seed=0).flatten(1)
    assert torch.allclose(m.min(1)[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(m.max(1)[0], torch.ones(2), atol=1e-4)


# ----------------------------------------------------------------------------- correctness
def test_no_noise_single_sample_equals_vanilla(model, x):
    t = torch.tensor([3, 7])
    a = smoothgrad(model, x, t, n_samples=1, noise_level=0.0)
    b = vanilla_gradient(model, x, t)
    assert torch.allclose(a, b, atol=1e-6)


def test_no_noise_many_samples_equals_vanilla(model, x):
    t = torch.tensor([3, 7])
    assert torch.allclose(smoothgrad(model, x, t, n_samples=4, noise_level=0.0),
                          vanilla_gradient(model, x, t), atol=1e-5)


def test_linear_model_gradient_is_the_weight_row_and_noise_is_irrelevant():
    """f(x) = W x is linear, so the gradient is W[c] for every input: SmoothGrad == vanilla for any noise."""
    torch.manual_seed(0)
    lin = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 5)).eval()
    xs = torch.randn(2, 3, 8, 8)
    t = torch.tensor([1, 4])
    g = input_gradient(lin, xs, t)
    W = lin[1].weight
    assert torch.allclose(g, W[t].view(2, 3, 8, 8), atol=1e-6)
    expected = minmax_normalise(W[t].view(2, 3, 8, 8).abs().amax(1))
    sg = smoothgrad(lin, xs, t, n_samples=6, noise_level=0.5, seed=0)
    assert torch.allclose(sg, expected, atol=1e-5)


def test_input_gradient_matches_finite_differences():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Tanh(), nn.Flatten(), nn.Linear(4 * 6 * 6, 3)).double().eval()
    xs = torch.randn(1, 3, 8, 8, dtype=torch.double)
    d = torch.randn_like(xs)
    eps = 1e-6
    with torch.no_grad():
        num = (net(xs + eps * d)[0, 2] - net(xs - eps * d)[0, 2]) / (2 * eps)
    ana = (input_gradient(net, xs, torch.tensor([2])) * d).sum()
    assert torch.allclose(num, ana, atol=1e-6)


def test_noise_std_is_proportional_to_the_input_range(model):
    """sigma = noise_level * (max - min): the same noise_level must perturb a x10-scaled image 10x more."""
    from smoothgrad import _noise_std
    a = torch.randn(1, 3, 16, 16)
    assert torch.allclose(_noise_std(10 * a, 0.15), 10 * _noise_std(a, 0.15), rtol=1e-5)
    assert torch.allclose(_noise_std(a, 0.15), 0.15 * (a.max() - a.min()), rtol=1e-5)


# ----------------------------------------------------------------------------- batching and targets
def test_batch_equals_single(model, x):
    batch = smoothgrad(model, x, None, n_samples=2, noise_level=0.0)
    single = torch.cat([smoothgrad(model, x[i:i + 1], None, n_samples=2, noise_level=0.0) for i in range(len(x))])
    assert torch.allclose(batch, single, atol=1e-4)


def test_none_int_list_and_tensor_targets_agree(model, x):
    pred = model(x).argmax(1)
    kw = dict(n_samples=2, noise_level=0.0)
    ref = smoothgrad(model, x, pred, **kw)
    assert torch.allclose(smoothgrad(model, x, None, **kw), ref)
    assert torch.allclose(smoothgrad(model, x, pred.tolist(), **kw), ref)
    a = smoothgrad(model, x, 5, **kw)
    b = smoothgrad(model, x, [5, 5], **kw)
    c = smoothgrad(model, x, torch.tensor([5, 5]), **kw)
    assert torch.allclose(a, b) and torch.allclose(b, c)


def test_different_targets_give_different_maps(model, x):
    kw = dict(n_samples=2, noise_level=0.0)
    assert not torch.allclose(smoothgrad(model, x[:1], 1, **kw), smoothgrad(model, x[:1], 2, **kw))


def test_wrong_target_length_raises(model, x):
    with pytest.raises(ValueError, match="target"):
        smoothgrad(model, x, [1, 2, 3], n_samples=1)


def test_none_target_is_fixed_on_the_clean_image(model, x):
    """With target=None all noisy copies must explain the clean image's class (not re-argmax per copy)."""
    pred = model(x).argmax(1)
    a = smoothgrad(model, x, None, n_samples=4, noise_level=0.3, seed=7)
    b = smoothgrad(model, x, pred, n_samples=4, noise_level=0.3, seed=7)
    assert torch.allclose(a, b)


# ----------------------------------------------------------------------------- randomness
def test_same_seed_is_reproducible_and_different_seed_differs(model, x):
    a = smoothgrad(model, x, None, n_samples=3, seed=1)
    b = smoothgrad(model, x, None, n_samples=3, seed=1)
    c = smoothgrad(model, x, None, n_samples=3, seed=2)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_seed_does_not_touch_global_rng(model, x):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    smoothgrad(model, x, None, n_samples=2, seed=5)
    assert torch.allclose(torch.rand(3), expected)


def test_more_samples_means_more_stable_maps(model, x):
    """Core claim of SmoothGrad: averaging reduces the sampling noise of the map."""
    def spread(n):
        maps = [smoothgrad(model, x[:1], 3, n_samples=n, noise_level=0.15, seed=s) for s in range(4)]
        return sum((maps[i] - maps[j]).abs().mean() for i in range(4) for j in range(i + 1, 4)) / 6
    assert spread(16) < spread(1)


# ----------------------------------------------------------------------------- side effects
def test_input_is_not_modified(model, x):
    x0 = x.clone()
    smoothgrad(model, x, None, n_samples=2, seed=0)
    assert torch.equal(x, x0) and not x.requires_grad


def test_parameter_grads_are_not_written(model, x):
    smoothgrad(model, x, None, n_samples=2, seed=0)
    assert all(p.grad is None for p in model.parameters())


def test_training_mode_is_restored():
    m = torchvision.models.resnet18(weights=None)
    m.train()
    smoothgrad(m, torch.randn(1, 3, 64, 64), 0, n_samples=1)
    assert m.training
    m.eval()
    smoothgrad(m, torch.randn(1, 3, 64, 64), 0, n_samples=1)
    assert not m.training


def test_works_with_frozen_parameters(x):
    m = torchvision.models.resnet18(weights=None).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    assert smoothgrad(m, x, None, n_samples=2, seed=0).shape == (2, 64, 64)


# ----------------------------------------------------------------------------- argument validation
@pytest.mark.parametrize("kwargs", [dict(n_samples=0), dict(n_samples=-3), dict(noise_level=-0.1),
                                    dict(mode="median"), dict(reduce="mean")])
def test_invalid_arguments_raise(model, x, kwargs):
    with pytest.raises(ValueError):
        smoothgrad(model, x, None, **kwargs)


# ----------------------------------------------------------------------------- smooth Grad-CAM
def test_smooth_gradcam_shape_range_and_hook_cleanup(model, x):
    n_before = len(model.layer4._forward_hooks)
    with GradCAM(model, model.layer4) as g:
        m = smooth_gradcam(g, x, None, n_samples=3, noise_level=0.1, seed=0)
    assert m.shape == (2, 64, 64) and m.min() >= 0 and m.max() <= 1 and not torch.isnan(m).any()
    assert len(model.layer4._forward_hooks) == n_before


def test_smooth_gradcam_without_noise_equals_gradcam(model, x):
    with GradCAM(model, model.layer4) as g:
        assert torch.allclose(smooth_gradcam(g, x, None, n_samples=2, noise_level=0.0), g(x), atol=1e-5)


def test_smooth_gradcam_is_reproducible_with_seed(model, x):
    with GradCAM(model, model.layer4) as g:
        a = smooth_gradcam(g, x, None, n_samples=2, noise_level=0.2, seed=3)
        b = smooth_gradcam(g, x, None, n_samples=2, noise_level=0.2, seed=3)
    assert torch.allclose(a, b)


# ----------------------------------------------------------------------------- percentile clipping
def test_clip_percentile_caps_outliers_and_keeps_range(model, x):
    kw = dict(n_samples=2, noise_level=0.0)
    raw = smoothgrad(model, x, 3, **kw)
    clipped = smoothgrad(model, x, 3, clip_percentile=99, **kw)
    assert clipped.min() >= 0 and clipped.max() <= 1 and not torch.isnan(clipped).any()
    # at least 1% of pixels are saturated at 1 after clipping, and the raw max was a lone outlier
    assert ((clipped.flatten(1) > 0.999).float().mean(1) >= 0.0099).all()
    assert (clipped.flatten(1).mean(1) > raw.flatten(1).mean(1)).all()   # clipping brightens the map


def test_clip_percentile_default_is_off_and_100_is_identity(model, x):
    kw = dict(n_samples=2, noise_level=0.0)
    assert torch.allclose(smoothgrad(model, x, 3, **kw), smoothgrad(model, x, 3, clip_percentile=100, **kw), atol=1e-6)


@pytest.mark.parametrize("p", [0, -5, 101])
def test_invalid_clip_percentile_raises(model, x, p):
    with pytest.raises(ValueError):
        smoothgrad(model, x, 3, n_samples=1, clip_percentile=p)