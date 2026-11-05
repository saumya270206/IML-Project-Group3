# Grad-CAM and SmoothGrad from scratch (PyTorch)

Both algorithms are implemented from scratch with plain PyTorch autograd. No XAI library
(captum, pytorch-grad-cam, ...) is used. torchvision is used only for the pretrained
classifier and image loading; matplotlib for plots.

| File | Content |
|---|---|
| `gradcam.py` | `GradCAM`: forward hook on the target conv layer, gradient-weighted channel averaging, ReLU, bilinear upsampling |
| `smoothgrad.py` | `vanilla_gradient`, `smoothgrad` (Gaussian noise, averaging over N samples), plus `smooth_gradcam` (extra) |
| `utils.py` | ImageNet-subset loader, preprocessing, pretrained model setup, overlay helpers, Deletion/Insertion metrics |
| `run_demo.py` | heatmap overlays, method comparison, quantitative evaluation, ablations |
| `requirements.txt` | dependencies |

## Install

```bash
git clone <your-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

## Data: an ImageNet subset

Put any images in a folder (searched recursively), e.g. ~100 ImageNet validation images, or the
[Imagenette](https://github.com/fastai/imagenette) / Kaggle "imagenet-mini" subsets (these are ImageNet classes):

```
data/imagenet_subset/
    n01440764/*.JPEG
    n02102040/*.JPEG
    ...          (a flat folder of .jpg/.png also works)
```
Labels are not needed: the methods explain the model's *predicted* class.

## Run

```bash
# full run (downloads pretrained ResNet-50 weights on first use)
python run_demo.py --data_dir ./data/imagenet_subset

# options
python run_demo.py --data_dir ./data/imagenet_subset --model vgg16 --n_eval 100 --n_samples 50 --noise 0.15

# offline smoke test (random weights + synthetic images, checks that the code runs)
python run_demo.py --fake --model resnet18 --n_eval 4 --n_demo 2 --n_abl 2 --n_samples 3
```

Everything is written to `./outputs`: `comparison.png`, `overlays/`, `ablation_noise.png`,
`ablation_samples.png`, `ablation_layer.png`, `metrics.json`, `metrics.md`.

## Methods

**Grad-CAM** (Selvaraju et al., 2017). For class `c` and conv activations `A` (K channels):
`alpha_k = mean_{i,j} dy^c/dA^k_ij`, `CAM = ReLU(sum_k alpha_k A^k)`, upsampled to the input size.
`y^c` is the pre-softmax score. Default layer: `layer4` (ResNet) or `features.29` (VGG16).

**SmoothGrad** (Smilkov et al., 2017). Average the gradient saliency over `N` noisy copies of the input,
`x + N(0, sigma^2)`, `sigma = noise_level * (x_max - x_min)`. Colour channels are collapsed with a max over |.|.
Modes: `mean` (paper), `abs`, `square`.

**Smooth Grad-CAM** (extra). Grad-CAM averaged over noisy inputs.

## Evaluation

No ground-truth explanation exists, so faithfulness is measured with the Deletion / Insertion
metrics of RISE (Petsiuk et al., 2018), computed on the predicted class probability:

* **Deletion AUC (lower is better)**: remove the most salient pixels first; a good map makes the probability collapse quickly.
* **Insertion AUC (higher is better)**: reveal the most salient pixels first on a blurred image.
* **Random** saliency is reported as a sanity baseline; any useful method should beat it.

`run_demo.py` prints a table and stores it in `outputs/metrics.md`. Paste your own numbers here after running:

| Method | Deletion ↓ | Insertion ↑ | Ins − Del ↑ |
|---|---|---|---|
| *(fill from outputs/metrics.md)* | | | |

## Ablations

* **Noise level** (`0 ... 0.5`): too little noise gives no smoothing, too much destroys the image.
* **Sample count** (`1 ... 50`): the map becomes smoother and the metrics stabilise as N grows.
* **Conv layer** for Grad-CAM (`layer1 ... layer4`): deeper layers are more class-specific but spatially coarser.

Skip them with `--skip_ablation`.

## Notes
* The model is always in `eval()` mode, so BatchNorm uses running statistics and batch samples are independent
  (this is what allows summing the class scores over a batch before differentiating).
* Results depend on the chosen images and seed (`--seed`); the ImageNet subset is a random sample.

## Tests and sanity checks (Grad-CAM)

```bash
python -m pytest tests -q
python gradcam_sanity.py --image <path>
python gradcam_sanity.py --image <path> --class_a "gas pump" --class_b "jinrikisha" --out outputs/sanity
```

Details, results and a log of the problems we hit are in [GRADCAM_NOTES.md](GRADCAM_NOTES.md).

**Windows:** if `run_demo.py` ends with `UnicodeEncodeError: 'charmap' codec can't encode character`,
set `$env:PYTHONUTF8=1` in PowerShell before running.

## References
* Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
* Smilkov et al., *SmoothGrad: removing noise by adding noise*, 2017.
* Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018.