# Grad-CAM and SmoothGrad from scratch (PyTorch)

Both algorithms are implemented from scratch with plain PyTorch autograd. No XAI library (captum, pytorch-grad-cam, ...) is used. torchvision is used only for the pretrained classifier and image loading; matplotlib for plots.

| File               | Content                                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------------------|
| `gradcam.py`       | `GradCAM`: forward hook on the target conv layer, gradient-weighted channel averaging, ReLU, bilinear upsampling |
| `smoothgrad.py`    | `vanilla_gradient`, `smoothgrad` (Gaussian noise, averaging over N samples), plus `smooth_gradcam` (extra)       |
| `utils.py`         | ImageNet-subset loader, preprocessing, pretrained model setup, overlay helpers, Deletion/Insertion metrics       |
| `run_demo.py`      | heatmap overlays, method comparison, quantitative evaluation, ablations                                          |
| `gradcam_sanity.py`| sanity checks for Grad-CAM (class discrimination, model randomisation)                                           |
| `tests/`           | unit tests for Grad-CAM                                                                                          |
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

On a CPU the full default run takes a long time (SmoothGrad dominates). A run on a MacBook Air (CPU) with `--n_eval 50 --n_demo 6 --n_abl 8 --n_samples 20` took on the order of an hour or more. Use smaller values for a quick check, or run on a GPU (e.g. a Kaggle notebook).

## Methods

**Grad-CAM** (Selvaraju et al., 2017). For class `c` and conv activations `A` (K channels): `alpha_k = mean_{i,j} dy^c/dA^k_ij`, `CAM = ReLU(sum_k alpha_k A^k)`, upsampled to the input size. `y^c` is the pre-softmax score. Default layer: `layer4` (ResNet) or `features.29` (VGG16).

**SmoothGrad** (Smilkov et al., 2017). Average the gradient saliency over `N` noisy copies of the input, `x + N(0, sigma^2)`, `sigma = noise_level * (x_max - x_min)`. Colour channels are collapsed with a max over |.|. Modes: `mean` (paper), `abs`, `square`.

**Smooth Grad-CAM** (extra). Grad-CAM averaged over noisy inputs.

## Evaluation

No ground-truth explanation exists, so faithfulness is measured with the Deletion / Insertion metrics of RISE (Petsiuk et al., 2018), computed on the predicted class probability:

- **Deletion AUC (lower is better)**: remove the most salient pixels first; a good map makes the probability collapse quickly.
- **Insertion AUC (higher is better)**: reveal the most salient pixels first on a blurred image.
- **Random** saliency is reported as a sanity baseline; any useful method should beat it.

`run_demo.py` prints a table and stores it in `outputs/metrics.md`.

## Results

Setup: ResNet-50 (pretrained, torchvision), Grad-CAM on `layer4`, 50 images from an ImageNet subset (seeded random sample, `--seed 0`), SmoothGrad with 20 samples and noise 0.15, CPU. Command:

```
python run_demo.py --data_dir ./data/imagenet_subset --n_eval 50 --n_demo 6 --n_abl 8 --n_samples 20
```

| Method            | Deletion ↓ | Insertion ↑ | Ins − Del ↑ | sec / 100 img (CPU) |
|-------------------|------------|-------------|-------------|---------------------|
| Random            | 0.1223     | 0.2306      | 0.1083      | 0.02                |
| Vanilla gradient  | 0.0811     | 0.3172      | 0.2360      | 13.50               |
| SmoothGrad (mean) | 0.0643     | 0.4568      | 0.3925      | 273.82              |
| SmoothGrad (abs)  | 0.0936     | 0.5432      | 0.4495      | 270.47              |
| Grad-CAM          | 0.1087     | 0.5890      | 0.4803      | 12.30               |
| Smooth Grad-CAM   | 0.1401     | 0.5390      | 0.3989      | 215.61              |

Per-image standard deviations are in `outputs/metrics.json`. Qualitative maps are in `outputs/comparison.png` and `outputs/overlays/`.

What the table shows:

- Every method beats Random on Insertion by a clear margin.
- Grad-CAM has the highest Insertion and the highest Insertion − Deletion, and is about 20x faster than SmoothGrad.
- SmoothGrad (mean) has the lowest (best) Deletion. SmoothGrad (abs) has the second-highest Insertion.
- Grad-CAM's Deletion is close to Random. A likely, untested reason is that it comes from a coarse 7x7 grid, so deleting part of a blob leaves much of the object visible.
- Smooth Grad-CAM does not improve on plain Grad-CAM here and costs much more time.
- With 50 images the differences between close methods (e.g. Grad-CAM vs SmoothGrad (abs) on Ins − Del) are small compared with the per-image spread and should not be over-interpreted.

## Ablations

Each ablation uses only 8 images (`--n_abl 8`), so only the large trends are reliable. Plots: `outputs/ablation_noise.png`, `outputs/ablation_samples.png`, `outputs/ablation_layer.png`.

**SmoothGrad noise level** (20 samples)

| Noise | Deletion ↓ | Insertion ↑ | Ins − Del ↑ |
|-------|------------|-------------|-------------|
| 0.0   | 0.0904     | 0.3512      | 0.2608      |
| 0.05  | 0.0743     | 0.4804      | 0.4061      |
| 0.1   | 0.0724     | 0.4900      | 0.4176      |
| 0.15  | 0.0725     | 0.5033      | 0.4308      |
| 0.2   | 0.0639     | 0.4880      | 0.4241      |
| 0.3   | 0.0808     | 0.3843      | 0.3035      |
| 0.5   | 0.0739     | 0.3276      | 0.2536      |

No noise gives the weakest maps (no smoothing), values around 0.1-0.2 work best, and too much noise (0.3-0.5) destroys the image and hurts Insertion.

**SmoothGrad sample count** (noise 0.15)

| N  | Deletion ↓ | Insertion ↑ | Ins − Del ↑ |
|----|------------|-------------|-------------|
| 1  | 0.0787     | 0.4621      | 0.3835      |
| 5  | 0.0667     | 0.4692      | 0.4025      |
| 10 | 0.0680     | 0.4923      | 0.4243      |
| 25 | 0.0772     | 0.5120      | 0.4349      |
| 50 | 0.0768     | 0.5270      | 0.4502      |

Insertion increases steadily with N, Deletion is roughly flat and noisy, and cost grows linearly with N.

**Grad-CAM target layer** (ResNet-50)

| Layer  | Deletion ↓ | Insertion ↑ | Ins − Del ↑ |
|--------|------------|-------------|-------------|
| layer1 | 0.1757     | 0.3264      | 0.1507      |
| layer2 | 0.1367     | 0.3324      | 0.1958      |
| layer3 | 0.1664     | 0.4100      | 0.2436      |
| layer4 | 0.1144     | 0.6928      | 0.5784      |

Deeper layers are more class-specific but spatially coarser. `layer4` is clearly the best choice by these metrics.

Skip the ablations with `--skip_ablation`.

## Notes

- The model is always in `eval()` mode, so BatchNorm uses running statistics and batch samples are independent (this is what allows summing the class scores over a batch before differentiating).
- Results depend on the chosen images and seed (`--seed`); the ImageNet subset is a random sample.

## Tests and sanity checks (Grad-CAM)

```
python -m pytest tests -q
python gradcam_sanity.py --image <path>
python gradcam_sanity.py --image <path> --class_a "gas pump" --class_b "jinrikisha" --out outputs/sanity
```

Details, results and a log of the problems we hit are in [GRADCAM_NOTES.md](GRADCAM_NOTES.md).

**Windows:** if `run_demo.py` ends with `UnicodeEncodeError: 'charmap' codec can't encode character`, set `$env:PYTHONUTF8=1` in PowerShell before running.

## References

- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Smilkov et al., *SmoothGrad: removing noise by adding noise*, 2017.
- Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018.
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
