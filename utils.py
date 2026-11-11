"""
Shared helpers: ImageNet-subset loading, preprocessing, pretrained-model setup,
visualisation helpers and the Deletion / Insertion faithfulness metrics.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_CONFIGS = {
    "resnet50": dict(weights="ResNet50_Weights.IMAGENET1K_V1", layer="layer4",
                     ablation_layers=["layer1", "layer2", "layer3", "layer4"]),
    "resnet18": dict(weights="ResNet18_Weights.IMAGENET1K_V1", layer="layer4",
                     ablation_layers=["layer1", "layer2", "layer3", "layer4"]),
    "vgg16": dict(weights="VGG16_Weights.IMAGENET1K_V1", layer="features.29",
                  ablation_layers=["features.8", "features.15", "features.22", "features.29"]),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- model
def load_model(name: str = "resnet50", pretrained: bool = True):
    """Returns (model in eval mode, list of 1000 ImageNet class names, config dict).
    pretrained=False gives random weights (offline smoke test only)."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"model must be one of {list(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[name]
    w = tvm.get_weight(cfg["weights"])
    model = getattr(tvm, name)(weights=w if pretrained else None)
    return model.eval(), w.meta["categories"], cfg


def get_layer(model, name: str):
    """Look up a sub-module by its dotted name, e.g. 'layer4' or 'features.29'."""
    return model.get_submodule(name)


# ----------------------------------------------------------------------------- data
def get_transform(size: int = 224):
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class ImageSubset(torch.utils.data.Dataset):
    """
    Any folder of images (searched recursively), e.g. an ImageNet subset laid out as
    <root>/<wnid>/*.JPEG or flat <root>/*.jpg. Labels are NOT needed: we explain the
    model's predicted class. A fixed random subset of `n` images is chosen (seeded).
    """
    def __init__(self, root, n=None, seed=0, size=224):
        paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not paths:
            raise FileNotFoundError(f"No images found under '{root}'. See README for how to get an ImageNet subset.")
        random.Random(seed).shuffle(paths)
        self.paths = paths[:n] if n else paths
        self.tf = get_transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            image = image.convert("RGB")
        return self.tf(image), str(self.paths[i])


class RandomImages(torch.utils.data.Dataset):
    """Synthetic blobby images for an offline smoke test (--fake)."""
    def __init__(self, n=8, seed=0, size=224):
        g = torch.Generator().manual_seed(seed)
        low = torch.rand(n, 3, 8, 8, generator=g)
        img = F.interpolate(low, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        self.x = TF.normalize(img, MEAN, STD)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], f"fake_{i}"


def load_all(ds):
    """Load a whole (small) dataset into one tensor + list of names."""
    xs, names = [], []
    for i in range(len(ds)):
        image, name = ds[i]
        xs.append(image)
        names.append(name)
    return torch.stack(xs), names


# ----------------------------------------------------------------------------- viz
def denormalize(x):
    """(B,3,H,W) normalised -> (B,3,H,W) in [0,1]."""
    m = torch.tensor(MEAN, device=x.device)[None, :, None, None]
    s = torch.tensor(STD, device=x.device)[None, :, None, None]
    return (x * s + m).clamp(0, 1)


def overlay(img, cam, alpha=0.5, cmap="jet"):
    """img: (H,W,3) float in [0,1]; cam: (H,W) in [0,1]. Returns blended (H,W,3)."""
    heat = matplotlib.colormaps[cmap](np.asarray(cam))[..., :3]
    return np.clip((1 - alpha) * np.asarray(img) + alpha * heat, 0, 1)


def minmax_normalise(m, eps=1e-8):
    """Per-sample min-max normalisation of a (B,H,W) map to [0,1]."""
    flat = m.flatten(1)
    lo = flat.amin(1, keepdim=True)
    hi = flat.amax(1, keepdim=True)
    return ((flat - lo) / (hi - lo + eps)).reshape_as(m)


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def deletion_insertion(model, x, sal, target, steps=32, blur_k=51, blur_sigma=11.0):
    """
    Faithfulness metrics of RISE (Petsiuk et al., 2018). Returns per-sample (deletion_auc, insertion_auc).
    Pixels are ranked by saliency (descending). At step t the top t/steps fraction of pixels is
      deletion : set to 0 (= dataset mean in normalised space)  -> class prob should drop fast (low AUC)
      insertion: copied from the image into a blurred copy       -> class prob should rise fast (high AUC)
    AUC = trapezoid area under the class-probability curve, x-axis = fraction of pixels in [0,1].
    """
    B, C, H, W = x.shape
    order = (sal.flatten(1) + 1e-6 * torch.rand_like(sal).flatten(1)).argsort(
        1, descending=True
    )
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(H * W, device=x.device).expand(B, -1))
    blurred = TF.gaussian_blur(x, [blur_k, blur_k], [blur_sigma, blur_sigma])
    zeros = torch.zeros_like(x)
    paired_targets = target.repeat(2)
    del_p, ins_p = [], []
    for t in range(steps + 1):
        mask = (rank < int(round(t * H * W / steps))).view(B, 1, H, W)
        if model.training:
            deletion_input = torch.where(mask, zeros, x)
            pd = F.softmax(model(deletion_input), 1).gather(1, target[:, None])[:, 0]
            insertion_input = torch.where(mask, x, blurred)
            pi = F.softmax(model(insertion_input), 1).gather(1, target[:, None])[:, 0]
        else:
            paired_inputs = x.new_empty((2 * B, C, H, W))
            paired_inputs[:B].copy_(torch.where(mask, zeros, x))
            paired_inputs[B:].copy_(torch.where(mask, x, blurred))
            probabilities = F.softmax(model(paired_inputs), 1).gather(
                1, paired_targets[:, None]
            )[:, 0]
            pd, pi = probabilities.split(B)
        del_p.append(pd)
        ins_p.append(pi)
    del_p, ins_p = torch.stack(del_p, 1), torch.stack(ins_p, 1)
    return torch.trapezoid(del_p, dx=1 / steps), torch.trapezoid(ins_p, dx=1 / steps)
