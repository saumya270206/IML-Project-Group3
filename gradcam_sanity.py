"""
Sanity checks for Grad-CAM on ONE image (run with pretrained weights).

 1) Class discrimination: maps for two different classes should highlight different regions.
 2) Model randomisation (Adebayo et al., 2018): re-initialise the network layer by layer, from the
    output backwards. A good explanation must CHANGE once the weights are destroyed; if the heatmap
    stays the same, it is explaining the input, not the model.

    python gradcam_sanity.py --image path/to/img.jpg
    python gradcam_sanity.py --image img.jpg --class_a "tabby" --class_b "golden retriever"
"""
import argparse, copy, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from gradcam import GradCAM
from utils import load_model, get_layer, get_transform, denormalize, overlay, set_seed


def class_index(classes, query):
    """int-like string -> index; otherwise first class whose name contains the text."""
    if str(query).isdigit():
        return int(query)
    hits = [i for i, c in enumerate(classes) if str(query).lower() in c.lower()]
    if not hits:
        raise SystemExit(f"no ImageNet class contains '{query}'")
    return hits[0]


def pearson(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def spearman(a, b):
    rank = lambda t: t.flatten().argsort().argsort().double()
    return pearson(rank(a), rank(b))


def randomise(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--layer", default=None)
    ap.add_argument("--class_a", default=None, help="index or name part; default = top-1 prediction")
    ap.add_argument("--class_b", default=None, help="index or name part; default = top-2 prediction")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="random weights (offline smoke test only)")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    model, classes, cfg = load_model(args.model, pretrained=not args.fake)
    layer_name = args.layer or cfg["layer"]
    x = get_transform()(Image.open(args.image).convert("RGB"))[None]
    img = denormalize(x)[0].permute(1, 2, 0).numpy()

    with torch.no_grad():
        top = model(x)[0].topk(2).indices.tolist()
    a = class_index(classes, args.class_a) if args.class_a is not None else top[0]
    b = class_index(classes, args.class_b) if args.class_b is not None else top[1]

    # ---------------- 1) class discrimination
    with GradCAM(model, get_layer(model, layer_name)) as g:
        cam_a, cam_b = g(x, a)[0], g(x, b)[0]
    print(f"\n[1] Class discrimination on layer '{layer_name}'")
    print(f"    A = {a} ({classes[a]}),  B = {b} ({classes[b]})")
    print(f"    Pearson r(map A, map B) = {pearson(cam_a, cam_b):.3f}   (closer to 1 = same region, lower = more class-specific)")
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for axis, im, t in [(ax[0], img, "Input"), (ax[1], overlay(img, cam_a.numpy()), f"A: {classes[a]}"),
                        (ax[2], overlay(img, cam_b.numpy()), f"B: {classes[b]}")]:
        axis.imshow(im); axis.set_title(t, fontsize=9); axis.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "sanity_class_discrimination.png"), dpi=150); plt.close()

    # ---------------- 2) cascading model randomisation (target class fixed to the ORIGINAL prediction)
    with GradCAM(model, get_layer(model, layer_name)) as g:
        ref = g(x, a)[0]
    rnd = copy.deepcopy(model)
    stages = [n for n, m in reversed(list(rnd.named_children())) if any(True for _ in m.parameters())]
    maps, labels, rows = [ref], ["trained"], []
    done = []
    for name in stages:
        randomise(getattr(rnd, name)); done.append(name)
        with GradCAM(rnd, get_layer(rnd, layer_name)) as g:
            m = g(x, a)[0]
        maps.append(m); labels.append(f"+ {name}")
        rows.append((name, spearman(ref, m), pearson(ref, m)))
    print(f"\n[2] Cascading randomisation (output -> input), target class fixed to {a} ({classes[a]})")
    print("    | randomised up to | Spearman vs trained | Pearson vs trained |\n    |---|---|---|")
    for n, s, p in rows:
        print(f"    | {n} | {s:.3f} | {p:.3f} |")
    print("    Expected: correlation drops towards 0 as more layers are randomised.\n")
    fig, ax = plt.subplots(1, len(maps), figsize=(2.2 * len(maps), 2.6))
    for axis, m, t in zip(ax, maps, labels):
        axis.imshow(overlay(img, m.numpy())); axis.set_title(t, fontsize=8); axis.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "sanity_randomisation.png"), dpi=150); plt.close()
    print(f"saved {args.out}/sanity_class_discrimination.png and {args.out}/sanity_randomisation.png")


if __name__ == "__main__":
    main()