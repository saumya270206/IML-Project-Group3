"""
Demo, quantitative evaluation and ablations for Grad-CAM and SmoothGrad on an ImageNet subset
with a pretrained torchvision classifier.

  python run_demo.py --data_dir ./data/imagenet_subset                 # full run
  python run_demo.py --fake --model resnet18 --n_eval 4 --n_demo 2 --n_abl 2 --n_samples 3   # offline smoke test

Outputs (in --out):
  comparison.png            heatmaps + overlays for several images and all methods
  overlays/*.png            Grad-CAM / Smooth Grad-CAM overlays for each demo image
  ablation_noise.png        SmoothGrad vs noise level   (maps + Deletion/Insertion)
  ablation_samples.png      SmoothGrad vs sample count
  ablation_layer.png        Grad-CAM vs conv layer
  metrics.json / metrics.md quantitative results (Deletion AUC lower=better, Insertion AUC higher=better)
"""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradcam import GradCAM
from smoothgrad import vanilla_gradient, smoothgrad, smooth_gradcam
from utils import (set_seed, load_model, get_layer, ImageSubset, RandomImages, load_all,
                   denormalize, overlay, deletion_insertion)


# ----------------------------------------------------------------------------- evaluation
def evaluate(methods, model, X, T_, args, dev):
    """methods: {name: fn(x, target) -> (B,H,W) saliency}. Returns {name: {del, ins, time}} (per-image lists)."""
    res = {k: {"del": [], "ins": [], "time": 0.0} for k in methods}
    for s in range(0, len(X), args.batch_size):
        xb, tb = X[s:s + args.batch_size].to(dev), T_[s:s + args.batch_size].to(dev)
        for k, fn in methods.items():
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            sal = fn(xb, tb).detach()
            if dev.type == "cuda": torch.cuda.synchronize()
            res[k]["time"] += time.time() - t0
            d, i = deletion_insertion(model, xb, sal, tb, steps=args.steps)
            res[k]["del"] += d.tolist(); res[k]["ins"] += i.tolist()
    return res


def summarise(res, n):
    out = {}
    for k, v in res.items():
        d, i = float(np.mean(v["del"])), float(np.mean(v["ins"]))
        out[k] = {"deletion_auc": d, "deletion_std": float(np.std(v["del"])),
                  "insertion_auc": i, "insertion_std": float(np.std(v["ins"])),
                  "ins_minus_del": i - d, "sec_per_100_imgs": v["time"] / n * 100}
    return out


def md_table(summary):
    lines = ["| Method | Deletion ↓ | Insertion ↑ | Ins − Del ↑ | sec / 100 img |", "|---|---|---|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {v['deletion_auc']:.4f} | {v['insertion_auc']:.4f} | "
                     f"{v['ins_minus_del']:.4f} | {v['sec_per_100_imgs']:.2f} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- method factories
def make_smoothgrad(model, n, nl, mode="mean"):
    return lambda x, t: smoothgrad(model, x, t, n_samples=n, noise_level=nl, mode=mode)


def make_smooth_gradcam(cam, n, nl):
    return lambda x, t: smooth_gradcam(cam, x, t, n_samples=n, noise_level=nl)


# ----------------------------------------------------------------------------- ablation
def run_ablation(title, labels, fns, model, X, T_, args, dev, out_png):
    """Evaluate one fn per setting; plot maps for the first image and Deletion/Insertion vs setting."""
    res = evaluate(dict(zip(labels, fns)), model, X, T_, args, dev)
    summ = summarise(res, len(X))
    x0, t0 = X[:1].to(dev), T_[:1].to(dev)
    n = len(labels)
    fig = plt.figure(figsize=(max(2.0 * n, 6), 6))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.1])
    for j, (lab, fn) in enumerate(zip(labels, fns)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(fn(x0, t0)[0].detach().cpu(), cmap="jet"); ax.set_title(lab, fontsize=8); ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(labels, [summ[l]["deletion_auc"] for l in labels], "o-", label="Deletion ↓")
    ax.plot(labels, [summ[l]["insertion_auc"] for l in labels], "s-", label="Insertion ↑")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close(fig)
    print(f"\n[{title}]\n" + md_table(summ))
    return summ


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data/imagenet_subset", help="folder of ImageNet images (recursive)")
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "resnet18", "vgg16"])
    ap.add_argument("--layer", default=None, help="Grad-CAM target layer (default depends on model)")
    ap.add_argument("--n_eval", type=int, default=64, help="images for quantitative evaluation")
    ap.add_argument("--n_demo", type=int, default=6, help="images in the qualitative figure")
    ap.add_argument("--n_abl", type=int, default=16, help="images for each ablation")
    ap.add_argument("--n_samples", type=int, default=25, help="SmoothGrad samples")
    ap.add_argument("--noise", type=float, default=0.15, help="SmoothGrad noise level (fraction of input range)")
    ap.add_argument("--steps", type=int, default=32, help="Deletion/Insertion steps")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--skip_ablation", action="store_true")
    ap.add_argument("--fake", action="store_true", help="offline smoke test: random weights + random images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    # ---- model + data
    model, classes, cfg = load_model(args.model, pretrained=not args.fake)
    model = model.to(dev).eval()
    n_total = max(args.n_eval, args.n_demo, args.n_abl)
    ds = RandomImages(n_total, args.seed) if args.fake else ImageSubset(args.data_dir, n_total, args.seed)
    X, names = load_all(ds)
    with torch.no_grad():
        T_ = torch.cat([model(X[s:s + args.batch_size].to(dev)).argmax(1).cpu()
                        for s in range(0, len(X), args.batch_size)])
    print(f"loaded {len(X)} images from {'synthetic data' if args.fake else args.data_dir}")

    layer_name = args.layer or cfg["layer"]
    cam = GradCAM(model, get_layer(model, layer_name))
    ns, nl = args.n_samples, args.noise

    methods = {
        "Random":            lambda x, t: torch.rand(x.shape[0], *x.shape[2:], device=x.device),
        "Vanilla gradient":  lambda x, t: vanilla_gradient(model, x, t),
        "SmoothGrad (mean)": make_smoothgrad(model, ns, nl, "mean"),
        "SmoothGrad (abs)":  make_smoothgrad(model, ns, nl, "abs"),
        "Grad-CAM":          lambda x, t: cam(x, t),
        "Smooth Grad-CAM":   make_smooth_gradcam(cam, ns, nl),
    }

    # ---- 1) qualitative comparison + overlays
    nd = args.n_demo
    xd, td = X[:nd].to(dev), T_[:nd].to(dev)
    maps = {k: methods[k](xd, td).detach().cpu() for k in
            ["Vanilla gradient", "SmoothGrad (mean)", "Grad-CAM", "Smooth Grad-CAM"]}
    img = denormalize(xd).cpu().permute(0, 2, 3, 1).numpy()
    cols = ["Input"] + list(maps) + ["Grad-CAM overlay", "Smooth Grad-CAM overlay"]
    fig, ax = plt.subplots(nd, len(cols), figsize=(2.3 * len(cols), 2.3 * nd), squeeze=False)
    for r in range(nd):
        ov_g, ov_s = overlay(img[r], maps["Grad-CAM"][r]), overlay(img[r], maps["Smooth Grad-CAM"][r])
        panels = [img[r]] + [m[r] for m in maps.values()] + [ov_g, ov_s]
        for c, p in enumerate(panels):
            ax[r, c].imshow(p, cmap=None if p.ndim == 3 else "jet"); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(cols[c], fontsize=8)
        ax[r, 0].set_title(f"pred: {classes[int(td[r])]}" if r else f"Input\npred: {classes[int(td[r])]}", fontsize=8)
        plt.imsave(os.path.join(args.out, "overlays", f"{r:02d}_gradcam.png"), ov_g)
        plt.imsave(os.path.join(args.out, "overlays", f"{r:02d}_smooth_gradcam.png"), ov_s)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "comparison.png"), dpi=130); plt.close(fig)
    print("saved comparison.png and overlays/")

    # ---- 2) quantitative evaluation
    ne = args.n_eval
    res = evaluate(methods, model, X[:ne], T_[:ne], args, dev)
    summary = summarise(res, ne)
    print(f"\nFaithfulness on {ne} images ({args.model}, layer={layer_name}, n={ns}, noise={nl})")
    print(md_table(summary))
    results = {"model": args.model, "layer": layer_name, "n_eval": ne, "n_samples": ns, "noise": nl,
               "fake": args.fake, "main": summary, "ablations": {}}
    md = [f"## Main results ({args.model}, {ne} images)", md_table(summary)]

    # ---- 3) ablations
    if not args.skip_ablation:
        na = args.n_abl
        Xa, Ta = X[:na], T_[:na]
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        s = run_ablation("SmoothGrad: noise level", [str(v) for v in noise_levels],
                         [make_smoothgrad(model, ns, v) for v in noise_levels], model, Xa, Ta, args, dev,
                         os.path.join(args.out, "ablation_noise.png"))
        results["ablations"]["noise_level"] = s; md += ["\n## Ablation: noise level", md_table(s)]

        counts = [1, 5, 10, 25, 50]
        s = run_ablation("SmoothGrad: number of samples", [str(v) for v in counts],
                         [make_smoothgrad(model, v, nl) for v in counts], model, Xa, Ta, args, dev,
                         os.path.join(args.out, "ablation_samples.png"))
        results["ablations"]["sample_count"] = s; md += ["\n## Ablation: sample count", md_table(s)]

        engines = [GradCAM(model, get_layer(model, l)) for l in cfg["ablation_layers"]]
        s = run_ablation("Grad-CAM: target conv layer", cfg["ablation_layers"],
                         [(lambda e: (lambda x, t: e(x, t)))(e) for e in engines], model, Xa, Ta, args, dev,
                         os.path.join(args.out, "ablation_layer.png"))
        for e in engines: e.remove()
        results["ablations"]["conv_layer"] = s; md += ["\n## Ablation: Grad-CAM layer", md_table(s)]

    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    open(os.path.join(args.out, "metrics.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\nAll outputs written to {args.out}")


if __name__ == "__main__":
    main()