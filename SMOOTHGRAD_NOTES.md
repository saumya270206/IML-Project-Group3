# SmoothGrad notes

Notes on the SmoothGrad part of the project: what `smoothgrad.py` does, how it was checked, what the results are, and what is still open.

## 1. What `smoothgrad.py` does

```
M(x)     = d y^c / d x                         (vanilla gradient, y^c = pre-softmax class score)
M_hat(x) = (1/n) * sum_i M(x + N(0, sigma^2))  sigma = noise_level * (x_max - x_min)   (per image)
```

Colour channels are collapsed with a max (or sum) over |.|, then the map is min-max normalised to [0,1].

| Function | Purpose |
|---|---|
| `input_gradient` | gradient of the class score wrt the input (autograd, no parameter `.grad` is written) |
| `vanilla_gradient` | baseline, equals SmoothGrad with `n_samples=1, noise_level=0` |
| `smoothgrad` | the method; `mode` = `mean` (paper), `abs`, `square`; `reduce` = `max` / `sum` |
| `smooth_gradcam` | extra: Grad-CAM averaged over noisy inputs |

Behaviour worth knowing:

* `target` can be `None` (predicted class of the **clean** image, fixed once), an `int`, a list or a `(B,)` tensor, same as `GradCAM`.
* `seed=` gives reproducible noise through a private `torch.Generator`; the global RNG is untouched.
* The model is put in `eval()` for the call and its previous mode is restored.
* `clip_percentile=99` (optional, default off) caps each map at its 99th percentile before normalising. The SmoothGrad paper does this for display, because a few extreme pixels otherwise make the whole map look dark. It does not change the pixel ranking, so Deletion/Insertion are unaffected (up to ties at the top).
* Invalid arguments (`n_samples < 1`, negative noise, unknown `mode`/`reduce`, bad percentile) raise `ValueError`.

## 2. Verification

`python -m pytest tests -q` (random weights, no download). `tests/test_smoothgrad.py` has 32 test cases:

| Test | What it checks |
|---|---|
| shape, range, no NaN | all `mode` x `reduce` combinations; each map reaches 0 and 1 |
| equals vanilla | `n=1, noise=0` and `n=4, noise=0` both give exactly the vanilla gradient |
| linear model | for `f(x) = Wx` the gradient is the weight row `W[c]`, and SmoothGrad equals vanilla for **any** noise (gradient is constant) |
| finite differences | `input_gradient` matches a numerical directional derivative (float64) |
| noise scale | `sigma` is proportional to the input range and equals `noise_level * (max - min)` |
| batch equals single | same maps for a batch and image by image (noise 0) |
| targets | `None`, int, list, tensor agree; different classes differ; wrong length raises; `None` is fixed on the clean image |
| seeds | same seed reproducible, different seed differs, global RNG untouched |
| variance reduction | maps from independent seeds agree more with `n=16` than with `n=1` (the core claim of the method) |
| side effects | input not modified, no parameter grads, train/eval mode restored, works with frozen parameters |
| validation | bad arguments raise `ValueError` |
| percentile clipping | range kept, outliers capped, 100 = identity, invalid values raise |
| Smooth Grad-CAM | shape/range, hook cleanup, equals Grad-CAM at noise 0, reproducible with a seed |

Status when this was written: all 40 tests in `tests/` pass (8 Grad-CAM + 32 SmoothGrad), on CPU.
`run_demo.py --fake ...` and `smoothgrad_sanity.py --fake ...` also ran end to end with the new code.

## 3. Sanity checks on real images (`smoothgrad_sanity.py`)

```bash
python smoothgrad_sanity.py --image <path> [--class_a "<n>" --class_b "<n>"] [--out outputs/sanity_sg]
```

1. **Run-to-run agreement**: Pearson/Spearman between two runs with different noise seeds, for n = 1, 5, 10, 25. Should rise with n.
2. **Class discrimination**: maps for two classes on the same image.
3. **Model randomisation** (Adebayo et al., 2018), same procedure as for Grad-CAM.
4. **Noise sweep figure**: noise 0 ... 0.5.

**Results on real images** (ResNet-50, pretrained, 1 image each, n = 25, noise 0.15; logs and figures in `outputs/sanity_sg_gaspump/` and `outputs/sanity_sg_parachute/`). Classes: gas station image = 571 (gas pump), parachute image = 701 (parachute).

| Image | n=1 | n=5 | n=10 | n=25 | Pearson r, class A vs B | Spearman after `fc` randomised |
|---|---|---|---|---|---|---|
| gas station | 0.221 | 0.321 | 0.414 | 0.554 | 0.543 (A = gas pump, B = jinrikisha) | 0.317 |
| parachute | 0.466 | 0.552 | 0.631 | 0.734 | 0.804 (A = parachute, B = balloon) | 0.572 |

The n columns are Pearson r between two runs with different noise seeds. Spearman between runs:

| Image | n=1 | n=5 | n=10 | n=25 |
|---|---|---|---|---|
| gas station | 0.223 | 0.260 | 0.329 | 0.421 |
| parachute | 0.615 | 0.632 | 0.662 | 0.699 |

Vanilla gradient vs SmoothGrad (n = 25), Pearson r: gas station -0.039, parachute -0.027.

Cascading randomisation (output -> input, target class fixed to A), Spearman / Pearson against the trained-model map:

| Randomised up to | gas station | parachute |
|---|---|---|
| `fc` | 0.317 / 0.357 | 0.572 / 0.561 |
| `layer4` | 0.151 / 0.173 | 0.149 / 0.234 |
| `layer3` | 0.087 / 0.042 | 0.103 / 0.056 |
| `layer2` | 0.088 / 0.029 | 0.138 / 0.044 |
| `layer1` | 0.117 / 0.044 | 0.183 / 0.074 |
| `bn1` | 0.127 / 0.050 | 0.213 / 0.093 |
| `conv1` | 0.103 / 0.029 | 0.187 / 0.067 |

What this shows:

* **Run-to-run agreement rises with n on both images, as the method predicts.** Pearson goes from 0.22 to 0.55 (gas station) and from 0.47 to 0.73 (parachute) between n = 1 and n = 25. Averaging removes sampling noise. It is still not 1 at n = 25 (0.55 on the gas station image), so two 25-sample maps from different seeds are similar but not identical; this matches the earlier finding that gains flatten around n = 25 (4.3).
* **Spearman rises more slowly than Pearson**, and on the parachute image it is already 0.615 at n = 1 (Pearson 0.466). So the pixel ranking is fairly stable even for a single noisy sample, while the large-magnitude pixels (which dominate Pearson) are what the averaging stabilises.
* **Vanilla gradient and SmoothGrad are almost uncorrelated** (r about -0.03 on both images). Pearson is dominated by a few extreme pixels, so this says the high-magnitude pixels of the raw gradient are not the ones that survive averaging. It does not by itself say which map is better; the Deletion/Insertion results in 4.1 favour SmoothGrad. Spearman for this pair was not computed.
* **Class discrimination is only partial.** The gas pump vs jinrikisha maps share a lot (r = 0.54), and parachute vs balloon even more (r = 0.80). Both pairs are plausible co-occurring classes in the same scene, and the sky/canopy region is relevant to both parachute and balloon, so a high value is not a failure, but the maps are not strongly class-specific. Only one class pair per image was tested.
* **Randomisation: the maps change clearly once the backbone is touched, but not after `fc` alone.** After randomising only `fc` the correlation stays fairly high (Spearman 0.32 gas station, 0.57 parachute; Pearson 0.36 and 0.56); after `layer4` it falls to 0.15-0.23, and from `layer3` down it stays at 0.09-0.21 (Spearman) and 0.03-0.09 (Pearson). A plausible, untested reason for the `fc` result is that the gradient still flows through the trained backbone, so the spatial pattern is only reweighted by the new `fc` row. The residual correlation at the bottom of the cascade is the edge structure that gradient maps keep for random weights (Adebayo et al., 2018). The parachute image keeps more (0.19 vs 0.10 at `conv1` in Spearman), which may be because its map follows strong, simple image edges.
* **Caveats:** one image per row, one seed pair, one randomisation draw, so no confidence intervals; differences of a few hundredths are within noise.

## 4. Quantitative results

Full run: `python run_demo.py --data_dir ./data/imagenet_subset` (ResNet-50, CPU, Deletion/Insertion AUC). Main table: **64 images**, 25 samples, noise 0.15. The ablations use only the **first 16 images** (`--n_abl` default), so they are noisier and must not be compared row by row with the main table.

### 4.1 Main results (64 images, n = 25, noise = 0.15)

| Method | Deletion (lower is better) | Insertion (higher is better) | Insertion - Deletion | seconds / 100 images |
|---|---|---|---|---|
| Random | 0.1081 | 0.2315 | 0.1234 | 0.05 |
| Vanilla gradient | 0.0733 | 0.3178 | 0.2445 | 23.00 |
| SmoothGrad (mean) | **0.0552** | 0.4592 | 0.4040 | 705.77 |
| SmoothGrad (abs) | 0.0763 | 0.5341 | 0.4578 | 540.18 |
| Grad-CAM | 0.1138 | **0.5904** | **0.4766** | 9.29 |
| Smooth Grad-CAM | 0.1419 | 0.5462 | 0.4044 | 261.08 |

* **SmoothGrad clearly beats the vanilla gradient.** Mean: Deletion 0.055 vs 0.073 and Insertion 0.459 vs 0.318. Abs: Insertion 0.534. Both are far above the random baseline.
* **SmoothGrad (mean) has the lowest Deletion of all methods** (0.0552): removing the pixels it ranks highest destroys the prediction fastest.
* **SmoothGrad (abs) is close to Grad-CAM on Insertion - Deletion** (0.458 vs 0.477). With one run and no confidence intervals, do not claim either is better.
* **Mean vs abs:** mean gives the best Deletion, abs the best Insertion. A plausible, untested reason is that averaging raw gradients lets positive and negative contributions cancel, which suppresses some pixels the model does use.
* **Grad-CAM's Deletion (0.114) is barely better than random (0.108)** even though its Insertion is the best. A plausible, untested reason is that its maps are coarse (7x7 upsampled), so it finds the right region but ranks pixels inside it poorly. Pixel-level methods do the opposite.
* **Smoothing does not help Grad-CAM here:** Smooth Grad-CAM is worse than plain Grad-CAM on both metrics (Ins - Del 0.404 vs 0.477).
* **Consistent with the earlier 20-image run** (n = 10): SmoothGrad (mean) had Deletion 0.0599 and Insertion 0.4504 there, so the picture did not change with more images and samples.
* **Cost:** 25 gradient passes per image, so roughly 25x the vanilla gradient (23 s x 25 is about 575 s; measured 540-706 s). Timings on a shared CPU varied by up to about 2x between runs (the n = 25, noise 0.15 setting took 318-340 s in the ablations), so use them as an order of magnitude only.

### 4.2 Ablation: noise level (16 images, n = 25)

| noise | Deletion | Insertion | Ins - Del |
|---|---|---|---|
| 0.0 | 0.0795 | 0.3377 | 0.2582 |
| 0.05 | 0.0718 | 0.4629 | 0.3911 |
| 0.1 | 0.0741 | 0.4718 | 0.3977 |
| 0.15 | 0.0745 | 0.4617 | 0.3872 |
| 0.2 | 0.0754 | 0.4374 | 0.3620 |
| 0.3 | 0.0716 | 0.4223 | 0.3506 |
| 0.5 | 0.0675 | 0.3495 | 0.2820 |

* **Noise is what makes SmoothGrad work.** With noise 0.0 all 25 samples are identical, so the result is just the vanilla gradient (Ins - Del 0.258). Adding even 5% noise raises it to 0.391.
* **Best range is about 0.05-0.15**, peaking at 0.1 (Ins - Del 0.398). It falls off at higher noise and is back near the no-noise level at 0.5, where the noisy image no longer resembles the original. This agrees with the 10-20% recommended in the paper.
* **The differences among 0.05, 0.1 and 0.15 are small** compared with run-to-run randomness (see 4.3), so the default 0.15 is fine; do not claim 0.1 is better.
* Deletion barely changes (0.068-0.075); the effect is almost all in Insertion.
* The noise-0.0 row differs a little from the vanilla row in 4.1 only because the ablation uses 16 images instead of 64.

### 4.3 Ablation: number of samples (16 images, noise = 0.15)

| n | Deletion | Insertion | Ins - Del | seconds / 100 images |
|---|---|---|---|---|
| 1 | 0.0694 | 0.4114 | 0.3419 | 15.23 |
| 5 | 0.0660 | 0.4350 | 0.3689 | 64.62 |
| 10 | 0.0689 | 0.4293 | 0.3604 | 160.04 |
| 25 | 0.0711 | 0.4612 | 0.3902 | 317.48 |
| 50 | 0.0816 | 0.4668 | 0.3852 | 605.67 |

* **Quality rises with n, then flattens around 25.** Going from 25 to 50 samples doubles the time and gives no gain (Ins - Del 0.390 to 0.385, Deletion slightly worse).
* **Even n = 5 gets most of the benefit** (0.369 vs 0.390) at about a fifth of the cost.
* **A single noisy sample (n = 1) already beats the noise-free gradient** (0.342 vs 0.258 in 4.2).
* **Run-to-run randomness:** n = 25, noise 0.15 appears in both ablations with different noise draws, giving Ins - Del 0.3902 and 0.3872. So differences of about 0.003 are noise, and the small ups and downs between n = 5, 10 and 25 should not be over-interpreted.

### 4.4 Limitations

* Single run, no confidence intervals; 16 images for the ablations.
* Deletion/Insertion are only one view of faithfulness, and they favour methods whose ranking suits the chosen pixel-removal baseline.
* Section 3 checks use one image per case and a single seed/randomisation draw, with one class pair per image.

## 5. Problems found and fixed

| # | Problem | Fix |
|---|---|---|
| 1 | `target=None` in `smooth_gradcam` re-picked the predicted class on every noisy copy, so copies could explain different classes | class fixed once on the clean image |
| 2 | `smoothgrad` / `input_gradient` accepted only a tensor target (int or list crashed) | shared `_resolve_target`, same forms as `GradCAM` |
| 3 | An unknown `mode` raised an obscure `KeyError`; `n_samples=0` gave a zero map or a division error | `ValueError` with a clear message |
| 4 | All three mode terms were computed on every iteration | only the requested one |
| 5 | The model was left in `eval()` after the call | previous mode restored |
| 6 | No way to reproduce a result | `seed=` with a private generator |
| 7 | Maps look dark because of a few extreme pixels | optional `clip_percentile` |

## References

* Smilkov et al., *SmoothGrad: removing noise by adding noise*, 2017.
* Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018.
* Petsiuk et al., *RISE: Randomized Input Sampling for Explanation of Black-box Models*, BMVC 2018.