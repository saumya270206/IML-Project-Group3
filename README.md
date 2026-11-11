# Grad-CAM and SmoothGrad from scratch (PyTorch)

Both algorithms are implemented from scratch with plain PyTorch autograd. No XAI library (captum, pytorch-grad-cam, ...) is used. torchvision is used only for the pretrained classifier and image loading; matplotlib for plots.

## Group members

| Name | Roll number | Contribution |
|------|-------------|--------------|
| Saumya Singh | 24AI10006 | Implemented SmoothGrad from scratch along with its 32 unit tests, ran sanity checks on real images and the noise and sample-count ablations, and documented the results. |
| Ishika Pajankar | 24AI10028 | Implemented and tested Grad-CAM, ran its sanity checks on real images, and documented the results and issues. |
| Yash Atul Dighade | 24AI10049 | Loaded ImageNet subset and set up ResNet-50 Pretrained model,holded overlay helpers and deletion/Insertion metrics |
| Abhishek Kumar | 24AI10007 | Ran the full evaluation on a GPU (Kaggle T4, 200 images) with the noise, sample-count and Grad-CAM layer ablations, produced the comparison figure and overlays, and wrote the README (install and run steps, results and ablation tables). |
| _TODO: name_ | _TODO: roll no._ | _TODO_ |

## Files

| File               | Content                                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------------------|
| `gradcam.py`       | `GradCAM`: forward hook on the target conv layer, gradient-weighted channel averaging, ReLU, bilinear upsampling |
| `smoothgrad.py`    | `vanilla_gradient`, `smoothgrad` (Gaussian noise, averaging over N samples), plus `smooth_gradcam` (extra)       |
| `utils.py`         | ImageNet-subset loader, preprocessing, pretrained model setup, overlay helpers, Deletion/Insertion metrics       |
| `run_demo.py`      | heatmap overlays, method comparison, quantitative evaluation, ablations                                          |
| `gradcam_sanity.py`| sanity checks for Grad-CAM (class discrimination, model randomisation)                                           |
| `smoothgrad_sanity.py` | sanity checks for SmoothGrad (run-to-run agreement vs N, class discrimination, model randomisation, noise sweep) |
| `tests/`           | unit tests for Grad-CAM (8) and SmoothGrad (32)                                                                  |
| `requirements.txt` | dependencies                                                                                                     |

## Install

```
git clone https://github.com/saumya270206/IML-Project-Group3.git
cd IML-Project-Group3
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No GPU is needed. The code uses CUDA if it is available and otherwise runs on the CPU (Apple-silicon Macs run on the CPU; MPS is not used). SmoothGrad and the Deletion/Insertion metrics are slow on a CPU, see the timings below.

## Data: an ImageNet subset

Put any images in a folder (searched recursively), e.g. ~100 ImageNet validation images, or the [Imagenette](https://github.com/fastai/imagenette) / Kaggle "imagenet-mini" subsets (these are ImageNet classes):

```
data/imagenet_subset/
    n01440764/*.JPEG
    n02102040/*.JPEG
    ...          (a flat folder of .jpg/.png also works)
```

Labels are not needed: the methods explain the model's *predicted* class.

Example with Imagenette (macOS/Linux):

```
mkdir -p data && cd data
curl -L -o imagenette.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette.tgz
mv imagenette2-160/val imagenet_subset
cd ..
```

## Run

```
# full run (downloads pretrained ResNet-50 weights on first use)
python run_demo.py --data_dir ./data/imagenet_subset

# options
python run_demo.py --data_dir ./data/imagenet_subset --model vgg16 --n_eval 100 --n_samples 50 --noise 0.15

# offline smoke test (random weights + synthetic images, checks that the code runs)
python run_demo.py --fake --model resnet18 --n_eval 4 --n_demo 2 --n_abl 2 --n_samples 3
```

Everything is written to `./outputs`: `comparison.png`, `overlays/`, `ablation_noise.png`, `ablation_samples.png`, `ablation_layer.png`, `metrics.json`, `metrics.md`.

On a CPU the full default run takes a long time (SmoothGrad dominates; a 50-image run on a MacBook Air CPU took roughly an hour). On a Kaggle T4 GPU the 200-image run below finished the quantitative part in about 12 minutes, so use a GPU (Kaggle or Colab) for the final numbers.

## Methods

**Grad-CAM** (Selvaraju et al., 2017). For class `c` and conv activations `A` (K channels): `alpha_k = mean_{i,j} dy^c/dA^k_ij`, `CAM = ReLU(sum_k alpha_k A^k)`, upsampled to the input size. `y^c` is the pre-softmax score. Default layer: `layer4` (ResNet) or `features.29` (VGG16).

**SmoothGrad** (Smilkov et al., 2017). Average the gradient saliency over `N` noisy copies of the input, `x + N(0, sigma^2)`, `sigma = noise_level * (x_max - x_min)`. Colour channels are collapsed with a max (or sum) over |.|. Modes: `mean` (paper), `abs`, `square`. `target` can be `None` (predicted class of the clean image), an int, a list or a tensor; `seed=` makes the noise reproducible without touching the global RNG; the optional `clip_percentile` (e.g. 99) caps extreme pixels before normalising, as in the paper, for display only.

**Smooth Grad-CAM** (extra). Grad-CAM averaged over noisy inputs.

## Evaluation

No ground-truth explanation exists, so faithfulness is measured with the Deletion / Insertion metrics of RISE (Petsiuk et al., 2018), computed on the predicted class probability:

- **Deletion AUC (lower is better)**: remove the most salient pixels first; a good map makes the probability collapse quickly.
- **Insertion AUC (higher is better)**: reveal the most salient pixels first on a blurred image.
- **Random** saliency is reported as a sanity baseline; any useful method should beat it.

`run_demo.py` prints a table and stores it in `outputs/metrics.md`.

## Results

`outputs_full/` holds an earlier 64-image CPU run (used in `SMOOTHGRAD_NOTES.md`), and `GRADCAM_NOTES.md` quotes a 20-image CPU run. The tables below come from the final 200-image GPU run stored in `outputs/`; the trends are the same in all three runs.

Setup: ResNet-50 (pretrained, torchvision), Grad-CAM on `layer4`, **200 images** from the Imagenette validation set (ImageNet classes; seeded random sample), SmoothGrad with 50 samples and noise 0.15, Kaggle T4 GPU. Command:

```
python run_demo.py --data_dir /tmp/imagenette2-160/val --n_eval 200 --n_demo 8 --n_abl 30 --n_samples 50 --noise 0.15
```

| Method            | Deletion ↓ | Insertion ↑ | Ins − Del ↑ | sec / 100 img (T4 GPU) |
|-------------------|------------|-------------|-------------|------------------------|
| Random            | 0.1119     | 0.2163      | 0.1044      | 0.00                   |
| Vanilla gradient  | 0.0861     | 0.2884      | 0.2023      | 0.76                   |
| SmoothGrad (mean) | 0.0601     | 0.4241      | 0.3641      | 37.91                  |
| SmoothGrad (abs)  | 0.0879     | 0.5011      | 0.4133      | 38.00                  |
| Grad-CAM          | 0.1133     | 0.5464      | 0.4331      | 0.37                   |
| Smooth Grad-CAM   | 0.1372     | 0.5080      | 0.3708      | 18.37                  |

Per-image standard deviations are in `outputs/metrics.json` (about 0.06-0.14 for Deletion and 0.18-0.29 for Insertion). Qualitative maps are in `outputs/comparison.png` and `outputs/overlays/`.

What the table shows:

- Every method beats Random on Insertion by a clear margin; the gradient-based maps also beat Random on Deletion, except Grad-CAM and Smooth Grad-CAM.
- Grad-CAM has the highest Insertion (0.546) and the highest Insertion − Deletion (0.433), and is about 100x faster than SmoothGrad on the GPU.
- SmoothGrad (mean) has the lowest (best) Deletion (0.060). SmoothGrad (abs) has the second-highest Insertion (0.501) and the second-highest Ins − Del (0.413).
- Grad-CAM's Deletion (0.113) is indistinguishable from Random (0.112). A likely, untested reason is that it comes from a coarse 7x7 grid, so deleting the top pixels removes a blob but leaves much of the object visible.
- Smooth Grad-CAM does not improve on plain Grad-CAM (Ins − Del 0.371 vs 0.433) and costs about 50x more time.
- The gap between Grad-CAM and SmoothGrad (abs) on Ins − Del (0.02) is small compared with the per-image spread, so those two should be treated as comparable; the larger gaps (both vs Vanilla and Random) are reliable.

## Ablations

Each ablation uses 30 images (`--n_abl 30`), so only the larger trends are reliable. Plots: `outputs/ablation_noise.png`, `outputs/ablation_samples.png`, `outputs/ablation_layer.png`.

**SmoothGrad noise level** (50 samples)

| Noise | Deletion ↓ | Insertion ↑ | Ins − Del ↑ |
|-------|------------|-------------|-------------|
| 0.0   | 0.0797     | 0.3069      | 0.2272      |
| 0.05  | 0.0785     | 0.4422      | 0.3637      |
| 0.1   | 0.0773     | 0.4443      | 0.3670      |
| 0.15  | 0.0705     | 0.4375      | 0.3670      |
| 0.2   | 0.0763     | 0.4331      | 0.3568      |
| 0.3   | 0.0774     | 0.4063      | 0.3290      |
| 0.5   | 0.0763     | 0.3244      | 0.2481      |

No noise (plain gradient) gives the weakest maps, there is a flat optimum for noise levels of about 0.05-0.2, and too much noise (0.3-0.5) destroys the image and hurts Insertion.

**SmoothGrad sample count** (noise 0.15)

| N  | Deletion ↓ | Insertion ↑ | Ins − Del ↑ |
|----|------------|-------------|-------------|
| 1  | 0.0705     | 0.3778      | 0.3074      |
| 5  | 0.0631     | 0.4012      | 0.3380      |
| 10 | 0.0612     | 0.4045      | 0.3433      |
| 25 | 0.0622     | 0.4192      | 0.3571      |
| 50 | 0.0733     | 0.4328      | 0.3595      |

Insertion and Ins − Del increase with N but with diminishing returns beyond about 25 samples, Deletion is roughly flat and noisy, and cost grows linearly with N.

**Grad-CAM target layer** (ResNet-50)

| Layer  | Deletion ↓ | Insertion ↑ | Ins − Del ↑ |
|--------|------------|-------------|-------------|
| layer1 | 0.1514     | 0.2429      | 0.0914      |
| layer2 | 0.1193     | 0.2506      | 0.1314      |
| layer3 | 0.1085     | 0.3523      | 0.2438      |
| layer4 | 0.1232     | 0.5934      | 0.4702      |

Deeper layers are more class-specific but spatially coarser. `layer4` is clearly the best choice by Insertion and Ins − Del, while `layer1` (Ins − Del 0.091) is no better than Random (0.104).

Skip the ablations with `--skip_ablation`.

## Notes

- The model is always in `eval()` mode, so BatchNorm uses running statistics and batch samples are independent (this is what allows summing the class scores over a batch before differentiating).
- Results depend on the chosen images and seed (`--seed`); the ImageNet subset is a random sample.

## Tests and sanity checks

```
python -m pytest tests -q                      # 40 tests, random weights, no download
```

**Grad-CAM**

```
python gradcam_sanity.py --image <path>
python gradcam_sanity.py --image <path> --class_a "gas pump" --class_b "jinrikisha" --out outputs/sanity
```

Details, results and a log of the problems we hit are in [GRADCAM_NOTES.md](GRADCAM_NOTES.md).

**SmoothGrad**

```
python smoothgrad_sanity.py --image <path>
python smoothgrad_sanity.py --image <path> --class_a "gas pump" --class_b "jinrikisha" --out outputs/sanity_sg
```

The script checks (1) run-to-run agreement between two noise seeds for N = 1, 5, 10, 25, (2) class discrimination, (3) cascading model randomisation (Adebayo et al., 2018) and (4) a noise-level sweep figure. On the gas station and parachute images (ResNet-50, N = 25, noise 0.15), agreement between runs rose with N (Pearson 0.22 to 0.55 and 0.47 to 0.73), and the maps changed clearly once the backbone was randomised, with a small residual correlation as expected for gradient maps. Class-specificity was only partial (r = 0.54 and 0.80 for the chosen class pairs). These are single-image checks, not statistics. Details, all tables, the SmoothGrad ablations and the problems found and fixed are in [SMOOTHGRAD_NOTES.md](SMOOTHGRAD_NOTES.md).

**Windows:** if `run_demo.py` ends with `UnicodeEncodeError: 'charmap' codec can't encode character`, set `$env:PYTHONUTF8=1` in PowerShell before running.

## References

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Smilkov et al., *SmoothGrad: removing noise by adding noise*, 2017.
- Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018.
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
