import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from gradcam import GradCAM
from utils import minmax_normalise


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None).eval()   # random weights: no download needed


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(2, 3, 224, 224)


def test_output_shape_and_range(model, x):
    with GradCAM(model, model.layer4) as g:
        cam = g(x)
    assert cam.shape == (2, 224, 224)
    assert cam.min() >= 0 and cam.max() <= 1
    assert not torch.isnan(cam).any()


def test_equals_cam_on_resnet(model, x):
    """For ResNet (layer4 -> GAP -> fc) Grad-CAM reduces exactly to CAM (Selvaraju et al., 2017)."""
    with GradCAM(model, model.layer4) as g:
        cam = g(x)
    feats = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: feats.update(a=o))
    with torch.no_grad():
        logits = model(x)
    h.remove()
    w = model.fc.weight[logits.argmax(1)]                       # (B, K)
    ref = F.relu((w[:, :, None, None] * feats["a"]).sum(1, keepdim=True))
    ref = F.interpolate(ref, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    assert torch.allclose(cam, minmax_normalise(ref), atol=1e-4)


def test_batch_equals_single(model, x):
    with GradCAM(model, model.layer4) as g:
        batch = g(x)
        single = torch.cat([g(x[i:i + 1]) for i in range(len(x))])
    assert torch.allclose(batch, single, atol=1e-4)


def test_int_list_and_tensor_targets_agree(model, x):
    with GradCAM(model, model.layer4) as g:
        a = g(x, 5)
        b = g(x, [5, 5])
        c = g(x, torch.tensor([5, 5]))
    assert torch.allclose(a, b) and torch.allclose(b, c)


def test_different_targets_give_different_maps(model, x):
    with GradCAM(model, model.layer4) as g:
        assert not torch.allclose(g(x[:1], 1), g(x[:1], 2))


def test_unused_layer_raises_clear_error(model, x):
    with GradCAM(model, nn.Conv2d(3, 3, 1)) as g:
        with pytest.raises(RuntimeError, match="not used"):
            g(x)


def test_hook_removed_after_with_block(model, x):
    n_before = len(model.layer4._forward_hooks)
    with GradCAM(model, model.layer4):
        assert len(model.layer4._forward_hooks) == n_before + 1
    assert len(model.layer4._forward_hooks) == n_before


def test_works_with_frozen_parameters(x):
    m = torchvision.models.resnet18(weights=None).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    with GradCAM(m, m.layer4) as g:
        assert g(x).shape == (2, 224, 224)