# Grad-CAM notes

Notes on the Grad-CAM part of the project: what was checked, what the results were, and which problems came up (with fixes). Setup used: ResNet-50 (pretrained, torchvision), target layer `layer4`, Windows, CPU only.

## 1. What `gradcam.py` does

For a class `c` and the activations `A` of the target conv layer:

```
alpha_k = mean over (i, j) of  d y^c / d A^k_ij      (y^c = pre-softmax class score)
CAM     = ReLU( sum_k alpha_k * A^k )                 (then bilinear upsampling + min-max normalisation)
```

* A forward hook stores the layer output; `torch.autograd.grad` gives the gradients of the class score with respect to it.
* The model is always put in `eval()` mode, so a batch can be processed at once (class scores are summed before differentiating).
* `target` can be `None` (predicted class), an `int`, a list of ints, or a `(B,)` tensor.
* The hook is removed with `with GradCAM(model, layer) as g:` or `g.remove()`.

## 2. Verification

Tests are in `tests/test_gradcam.py` (`python -m pytest tests -q`, random weights, no download needed):

| Test | What it checks |
|---|---|
| output shape and range | `(B,H,W)`, values in [0,1], no NaN |
| equals CAM on ResNet | For ResNet (`layer4` then global average pool then `fc`), Grad-CAM reduces exactly to CAM (Selvaraju et al., 2017). The maps match the CAM computed directly from the `fc` weights |
| batch equals single | Processing images together or one by one gives the same maps |
| int / list / tensor targets | All three forms give the same result |
| different targets differ | Different classes give different maps |
| unused layer | A clear `RuntimeError` is raised |
| hook removal | The hook is gone after the `with` block |
| frozen parameters | Works when `requires_grad` is off on all parameters |

Also checked by hand: on VGG16, hooking the last conv (`features[28]`) and its in-place ReLU (`features[29]`) gives identical maps.

## 3. Sanity checks on real images (`gradcam_sanity.py`)

```bash
python gradcam_sanity.py --image <path> [--class_a "<name>" --class_b "<name>"] [--out <folder>]
```

1. **Class discrimination**: maps for two different classes on the same image.
2. **Model randomisation** (Adebayo et al., 2018): the network is re-initialised from the output layer backwards; the map for the original predicted class is compared with the trained one (Spearman and Pearson correlation).

| Image | Classes compared | Pearson r between the two maps |
|---|---|---|
| gas station (`n03425413_8412`) | gas pump vs jinrikisha | 0.352 |
| parachute (`n03888257_7960`) | parachute vs balloon | 0.851 |

* Visually different classes give clearly different maps (r = 0.35). Visually similar classes (parachute vs balloon, both a round canopy in the sky) overlap strongly (r = 0.85).
* Randomisation: once the final `fc` layer is re-initialised, the correlation with the trained map falls to about zero and stays near zero for the later stages (parachute: Spearman -0.059 after `fc`; gas station: -0.261). So the map depends on the learned weights. This is a check on two images with one seed, not a statistical test; individual values fluctuate (for example Spearman 0.564 after `layer1` on the gas station image).

## 4. Quantitative results

`python run_demo.py --data_dir ./data/imagenet_subset --n_eval 20 --n_demo 4 --n_samples 10 --skip_ablation`
ResNet-50, `layer4`, 20 images, SmoothGrad with 10 samples and noise 0.15. Mean over images; standard deviations are in `outputs/metrics.json`.

| Method | Deletion (lower is better) | Insertion (higher is better) | Insertion - Deletion | seconds / 100 images (CPU) |
|---|---|---|---|---|
| Random | 0.1173 | 0.2481 | 0.1308 | 0.03 |
| Vanilla gradient | 0.0731 | 0.3233 | 0.2502 | 13.03 |
| SmoothGrad (mean) | 0.0599 | 0.4504 | 0.3905 | 122.77 |
| SmoothGrad (abs) | 0.0760 | 0.5195 | 0.4436 | 126.37 |
| Grad-CAM | 0.1493 | 0.5820 | 0.4327 | 5.09 |
| Smooth Grad-CAM | 0.1713 | 0.5323 | 0.3609 | 58.05 |

How to read it:

* All methods beat random on Insertion by a wide margin. Grad-CAM has the highest Insertion score.
* The sample is small (20 images) and the standard deviations are large (for example Grad-CAM Deletion 0.149 +- 0.176), so small gaps are within noise. Grad-CAM vs SmoothGrad (abs) on Insertion minus Deletion (0.433 vs 0.444) is not a real difference, and Grad-CAM's Deletion is not distinguishable from random here.
* Grad-CAM is about 25 times faster than SmoothGrad (5 s vs about 125 s per 100 images).
* Qualitatively (`outputs/comparison.png`), Grad-CAM maps land on the object (parachutes, the open rear of a garbage truck). Grad-CAM is coarse, since it comes from a 7x7 grid. A likely, untested reason for its weak Deletion score is that removing a part of a coarse blob leaves much of the object visible.
* A wrong prediction is also visible: the gas station image was classified as "jinrikisha" and the map highlights the red pump and the lower scene.

The layer ablation (`layer1` to `layer4`) was not run for this note.

## 5. Problems that came up and how they were fixed

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `TypeError: 'int' object is not subscriptable` when calling `GradCAM(...)(x, 5)` | `target` was assumed to be a tensor | `target` is converted with `torch.as_tensor(...)`; int, list and tensor all work |
| 2 | `TypeError: 'NoneType' object is not iterable` when the target layer is not part of the forward pass | The hook never fired, so no activations were stored | Explicit check that raises `RuntimeError("target_layer produced no output...")` |
| 3 | Hook stayed attached to the model unless `remove()` was called | No context manager | Added `__enter__` / `__exit__`: `with GradCAM(model, layer) as g:` |
| 4 | `UnicodeEncodeError: 'charmap' codec can't encode character '\u2193'` at the very end of `run_demo.py` on Windows, after the whole evaluation had finished | `metrics.md` was written with the Windows default encoding (cp1252), which cannot store the arrows in the table header | `open(..., "w", encoding="utf-8")` in `run_demo.py`. Alternative without editing code: `$env:PYTHONUTF8=1` in PowerShell before running |
| 5 | Pretrained-weights download was interrupted (`KeyboardInterrupt`) | The run was cancelled while the 98 MB ResNet-50 file was downloading | Run again and wait; the weights are cached afterwards in `~/.cache/torch/hub/checkpoints/` |
| 6 | The demo is slow on CPU | SmoothGrad needs many forward/backward passes, and Deletion/Insertion needs many forward passes per image | For a quick check use smaller values, e.g. `--n_eval 20 --n_demo 4 --n_samples 10 --skip_ablation`; leave the full run with ablations for last and do not interrupt it |

Already handled in the baseline: an all-zero Grad-CAM map (when the ReLU removes everything) does not give NaN, because `minmax_normalise` uses an epsilon.

## References

* Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
* Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
* Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018.
