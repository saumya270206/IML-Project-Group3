"""
Sanity checks for SmoothGrad on ONE image (run with pretrained weights).

 1) Noise reduction: two SmoothGrad runs with different noise seeds should agree more and more as the
    number of samples n grows (this is the whole point of the method). Reported as Pearson r between
    runs, vanilla gradient (n=1, no noise) is deterministic and shown for reference.
 2) Class discrimination: maps for two different classes should highlight different pixels.
 3) Model randomisation (Adebayo et al., 2018): re-initialise the network layer by layer, from the
    output backwards. A good explanation must CHANGE once the weights are destroyed.
 4) Noise-level sweep figure (sigma = noise_level * (x_max - x_min)).

    python smoothgrad_sanity.py --image path/to/img.jpg
    python smoothgrad_sanity.py --image img.jpg --class_a "gas pump" --class_b "jinrikisha" --out outputs/sanity_sg
    python smoothgrad_sanity.py --image img.jpg --fake --model resnet18 --n_samples 3     # offline smoke test
"""
import argparse, copy, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from gradcam_sanity import class_index, pearson, spearman, randomise     # tiny shared helpers
from smoothgrad import vanilla_gradient, smoothgrad
from utils import load_model, get_transform, denormalize, overlay, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--class_a", default=None, help="index or name part; default = top-1 prediction")
    ap.add_argument("--class_b", default=None, help="index or name part; default = top-2 prediction")
    ap.add_argument("--n_samples", type=int, default=25)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="random weights (offline smoke test only)")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    model, classes, _ = load_model(args.model, pretrained=not args.fake)
    x = get_transform()(Image.open(args.image).convert("RGB"))[None]
    img = denormalize(x)[0].permute(1, 2, 0).numpy()
    with torch.no_grad():
        top = model(x)[0].topk(2).indices.tolist()
    a = class_index(classes, args.class_a) if args.class_a is not None else top[0]
    b = class_index(classes, args.class_b) if args.class_b is not None else top[1]
    ta = torch.tensor([a])
    sg = lambda m, t, **kw: smoothgrad(m, x, t, n_samples=kw.pop("n", args.n_samples),
                                       noise_level=kw.pop("nl", args.noise), **kw)[0]

    # ---------------- 1) noise reduction: agreement between independent runs vs number of samples
    ns = sorted({1, 5, 10, args.n_samples})
    print(f"\n[1] Run-to-run agreement (two noise seeds), class {a} ({classes[a]}), noise={args.noise}")
    print("    | n samples | Pearson r between runs | Spearman between runs |\n    |---|---|---|")
    maps1 = []
    for n in ns:
        m1, m2 = sg(model, ta, n=n, seed=1000), sg(model, ta, n=n, seed=2000)
        print(f"    | {n} | {pearson(m1, m2):.3f} | {spearman(m1, m2):.3f} |")
        maps1.append(m1)
    print("    Expected: agreement increases with n (averaging removes the sampling noise).")
    van = vanilla_gradient(model, x, ta)[0]
    print(f"    Vanilla gradient vs SmoothGrad(n={args.n_samples}): Pearson r = {pearson(van, maps1[-1]):.3f}")

    # ---------------- 2) class discrimination
    ma, mb = sg(model, ta, seed=1), sg(model, torch.tensor([b]), seed=1)
    print(f"\n[2] Class discrimination: A = {a} ({classes[a]}),  B = {b} ({classes[b]})")
    print(f"    Pearson r(map A, map B) = {pearson(ma, mb):.3f}   (closer to 1 = same pixels, lower = more class-specific)")
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for axis, im, t, cm in [(ax[0], img, "Input", None), (ax[1], ma.numpy(), f"A: {classes[a]}", "jet"),
                            (ax[2], mb.numpy(), f"B: {classes[b]}", "jet")]:
        axis.imshow(im, cmap=cm); axis.set_title(t, fontsize=9); axis.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "sg_sanity_class_discrimination.png"), dpi=150); plt.close()

    # ---------------- 3) cascading model randomisation (target class fixed to the ORIGINAL class a)
    ref = sg(model, ta, seed=1)
    rnd = copy.deepcopy(model)
    stages = [n for n, m in reversed(list(rnd.named_children())) if any(True for _ in m.parameters())]
    maps, labels, rows = [ref], ["trained"], []
    for name in stages:
        randomise(getattr(rnd, name))
        m = sg(rnd, ta, seed=1)
        maps.append(m); labels.append(f"+ {name}")
        rows.append((name, spearman(ref, m), pearson(ref, m)))
    print(f"\n[3] Cascading randomisation (output -> input), target class fixed to {a} ({classes[a]})")
    print("    | randomised up to | Spearman vs trained | Pearson vs trained |\n    |---|---|---|")
    for n, s, p in rows:
        print(f"    | {n} | {s:.3f} | {p:.3f} |")
    print("    Expected: correlation drops towards 0 as more layers are randomised.")
    print("    Note: gradient maps keep some input-edge structure even for random weights, so a residual")
    print("    correlation is normal (Adebayo et al., 2018); what matters is the clear drop.\n")
    fig, ax = plt.subplots(1, len(maps), figsize=(2.2 * len(maps), 2.6), squeeze=False)
    for axis, m, t in zip(ax[0], maps, labels):
        axis.imshow(m.numpy(), cmap="jet"); axis.set_title(t, fontsize=8); axis.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "sg_sanity_randomisation.png"), dpi=150); plt.close()

    # ---------------- 4) noise-level sweep
    levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    fig, ax = plt.subplots(1, len(levels) + 1, figsize=(2.2 * (len(levels) + 1), 2.6))
    ax[0].imshow(img); ax[0].set_title("Input", fontsize=8); ax[0].axis("off")
    for axis, nl in zip(ax[1:], levels):
        axis.imshow(sg(model, ta, nl=nl, seed=1).numpy(), cmap="jet")
        axis.set_title(f"noise {nl}", fontsize=8); axis.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "sg_sanity_noise_sweep.png"), dpi=150); plt.close()
    print(f"saved sg_sanity_class_discrimination.png, sg_sanity_randomisation.png, sg_sanity_noise_sweep.png in {args.out}")


if __name__ == "__main__":
    main()