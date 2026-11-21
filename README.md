# Feature Lab

A desktop app (Python + PySide6) for **feature detection, description and matching**,
with every algorithm implemented from scratch:

| Step | Algorithm | Owner |
|---|---|---|
| 1 | Harris corner detector | Member 1 |
| 2 | λ− (minimum-eigenvalue) detector | Member 2 |
| 3 | SIFT 128-D descriptor | Member 3 |
| 4 | Matching with SSD and NCC | Member 4 |

Each step reports its **computation time**, and a Report tab collects all timings (CSV export).

---

## 1. Install and run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

In the app: **Load samples** (or *Add images* / drag-and-drop your own) → **▶ Run full pipeline** (F5) → open **Report**.

Without the GUI: `python run_cli.py samples/*.png --csv timings.csv`
Automated check: `pytest -q`

> OpenCV is used only for reading files, the convolution primitive and drawing.
> Sobel kernels, Gaussian kernels, Harris, λ−, NMS, SIFT, SSD and NCC are all our own code.

---

## 2. Project structure (who owns what)

```
feature_lab/
├── main.py                      app entry point
├── run_cli.py                   same pipeline without GUI
├── requirements.txt
├── features/                    ALGORITHMS (no GUI code)
│   ├── core/                    ── Member 1 (shared by everyone)
│   │   ├── image.py             load image, color → grayscale
│   │   ├── gradients.py         Gaussian, Sobel Ix/Iy, structure tensor M
│   │   ├── nms.py               threshold + non-maximum suppression
│   │   ├── keypoints.py         Keypoints container (the shared "contract")
│   │   ├── context.py           per-image cache of gradients / tensor
│   │   ├── params.py            param(...) helper → GUI forms built automatically
│   │   └── timer.py             Timer (ms)
│   ├── detectors/
│   │   ├── base.py              ── Member 1: FeatureDetector template
│   │   ├── harris.py            ── Member 1
│   │   └── lambda_minus.py      ── Member 2
│   ├── descriptors/sift.py      ── Member 3
│   ├── matching/matchers.py     ── Member 4
│   ├── visualization.py         drawing helpers (keypoints, heatmap, matches, descriptor)
│   └── pipeline.py              ── Member 1: FeatureProject (glues steps, caching, report)
├── gui/                         DESKTOP APP (no algorithm code)
│   ├── main_window.py           window, image list, toolbar, background jobs
│   ├── tabs.py                  one tab per step + Report tab
│   ├── widgets.py               image viewer, stat cards, auto parameter form
│   ├── worker.py                background thread
│   └── style.py                 theme
├── samples/                     demo images (+ known transforms for testing)
├── tools/make_samples.py        regenerates the demo images
└── tests/test_pipeline.py       correctness checks
```

---

## 3. How the pieces fit (and why nothing is duplicated)

```
          ImageData (load once: rgb for display, gray float for algorithms)
                │
          ImageContext  ── cache ──►  ImageGradients (Ix, Iy, magnitude, orientation)
                │                          │
                │                     StructureTensor (Sxx, Sxy, Syy)
                │                          │
     ┌──────────┴──────────┐               │
 HarrisDetector     LambdaMinusDetector  ◄─┘   both inherit FeatureDetector.detect():
  R = det − k·tr²     R = λ−                    tensor → response → threshold + NMS
     └──────────┬──────────┘
            Keypoints
                │
         SIFTDescriptor  (reuses the cached magnitude/orientation)
                │
          (N × 128) descriptors
                │
      SSDMatcher / NCCMatcher  (both inherit FeatureMatcher.match())
                │
           MatchResult  → drawing + timing report
```

OOP ideas used (easy to name in the discussion):

* **Inheritance + Template Method** — `FeatureDetector.detect()` and `FeatureMatcher.match()` contain the
  shared steps; a subclass only writes its own formula (`compute_response`, `score_matrix`, `accept`).
* **Caching / single responsibility** — `ImageContext` computes gradients and the structure tensor once;
  Harris, λ− and SIFT read them. Changing a parameter only recomputes what depends on it.
* **Registry** — `DETECTORS` and `MATCHERS` dictionaries; the GUI lists whatever is registered.
* **Separation of concerns** — `features/` has no GUI code, `gui/` has no algorithm code.
* **Parameter dataclasses** — each algorithm declares its parameters once with `param(...)`;
  the GUI builds the input form from them automatically.

How timing is reported (Report tab / CSV):

* **Detection** = gradients + structure tensor + response + threshold/NMS
  (the shared steps are included so each detector's number is the full cost from the raw image).
* **Description** = orientation assignment + descriptor building.
* **Matching** = score matrix + best-match selection + acceptance test + cross-check.

---

## 4. Discussion guide (one section per member)

### Member 1 — Harris + project setup
**Files:** `core/image.py`, `core/gradients.py`, `core/nms.py`, `detectors/base.py`, `detectors/harris.py`, `pipeline.py`

1. **Preprocessing:** color → gray with `Y = 0.299R + 0.587G + 0.114B`, scaled to [0, 1]. A 3-channel image
   whose channels are identical is detected as grayscale.
2. **Smoothing:** a Gaussian (σ = *Gradient smoothing σ*) removes noise before taking derivatives.
3. **Gradients:** Sobel kernels `[[-1,0,1],[-2,0,2],[-1,0,1]]` (x) and its transpose (y) → `Ix`, `Iy`.
4. **Structure tensor:** `M = G_σ * [[Ix², IxIy],[IxIy, Iy²]]` (σ = *Window σ*).
5. **Response:** `R = det(M) − k·trace(M)²`, k ≈ 0.04–0.06.
   R > 0 large → corner, R < 0 → edge, |R| small → flat.
6. **Threshold:** keep `R > ratio · max(R)`.
7. **NMS:** a pixel survives only if it equals the maximum of its window (`R == dilate(R)`),
   then keep the strongest *Max keypoints*.

Likely questions: *Why blur before Sobel?* (derivatives amplify noise). *Effect of k?* (bigger k → fewer
corners, more edge rejection). *Why NMS?* (one corner gives a blob of high R; we want one point).

### Member 2 — λ− detector
**File:** `detectors/lambda_minus.py` (reuses Member 1's tensor, threshold and NMS)

* M is symmetric 2×2 → closed-form eigenvalues for all pixels at once:
  `λ± = (a+c)/2 ± sqrt(((a−c)/2)² + b²)` with a = Sxx, b = Sxy, c = Syy.
* Response = **λ−** (smaller eigenvalue). Large λ− ⇒ strong change in *every* direction ⇒ corner.
* Compared with Harris: no constant k; λ− is linear in contrast while Harris R is quadratic, so λ− uses a
  higher default relative threshold (0.05).
* Use the **Response map** view to compare it with Harris on the same image.

### Member 3 — SIFT descriptors
**File:** `descriptors/sift.py`

1. **Orientation:** 36-bin histogram of gradient angles around the keypoint, weighted by magnitude ×
   Gaussian; peak refined with a parabola → θ.
2. **Rotate** the 16×16 sampling grid by −θ (rotation invariance).
3. **4×4 cells**, each with an **8-bin** histogram of angles relative to θ, weighted by magnitude ×
   Gaussian (σ = half the patch) → 4·4·8 = **128** values.
4. **Normalise → clip at 0.2 → normalise** (illumination robustness).
5. All keypoints are processed together with NumPy; one `np.bincount` builds every histogram.

Use *Inspect one descriptor* in the SIFT tab to show the 4×4×8 arrows of any keypoint.
Note: our detectors are single-scale, so the patch size is fixed (no scale-space pyramid).

### Member 4 — SSD and NCC matching
**File:** `matching/matchers.py`

* **SSD:** `‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b` → the whole N×M matrix is one matrix product. Best = minimum.
  Accepted if `best < ratio² · second best` (Lowe ratio, defined on distance, so squared for SSD).
* **NCC:** subtract each descriptor's mean, divide by its norm, then one matrix product gives all
  correlations in [−1, 1]. Best = maximum; accepted if ≥ *Min NCC*.
* **Cross-check:** keep i↔j only if j's best match is also i.
* The view draws the N most confident matches with coloured lines.

---

## 5. Team workflow

* Each member edits **only their own file(s)**; shared code lives in `features/core/` (Member 1).
* New parameters: add a `param(...)` field to your dataclass — it appears in the GUI automatically.
* Before pushing: `pytest -q` and `python run_cli.py samples/*.png`.
