## Grid Search Plan v2 – Spectra Best-Fit & Efficiency

This document refines the original `grid_search_plan.md` using insights from the proprietary LTA software’s interpretation history (as captured in `Spectra_Quality_Components_Analysis.md`) and general best practices for spectral fitting and grid search efficiency.

The goals are:

- **Metric alignment**: make the composite score reflect how the LTA tool visually judges “good” vs “bad” fits (amplitude/residual behaviour, fringe structure, and stability).
- **Search efficiency**: minimise the number of evaluated spectra while still reliably finding the same best-fit region the clinician would pick.

This plan describes **what** should be implemented and **where**, without forcing a particular low-level optimisation strategy.

---

## 1. Recap of Current Grid Search Behaviour

- **Driver**: `run_grid_search.py` orchestrates measurement loading, parameter enumeration, scoring, and saving top‑K results.
- **Preprocessing**: `src/analysis/measurement_utils.py` prepares:
  - detrended spectra (high‑pass filtered),
  - peak tables on the detrended signal,
  - resampled spectra and FFTs for phase analysis.
- **Metrics**: `src/analysis/metrics.py` computes:
  - `peak_count_score`: agreement in the number of peaks within a wavelength tolerance,
  - `peak_delta_score`: closeness of matched peak positions with penalties for unmatched peaks,
  - `phase_overlap_score`: FFT-based similarity of detrended spectra,
  - `composite_score`: weighted sum of the above metrics.
- **Search space**: `run_grid_search.py` currently performs a **single-pass exhaustive grid** over `(lipid_nm, aqueous_nm, roughness_A)` with regular spacing configured in `config.yaml`.
- **Output**: a ranked table (CSV/JSON) of the top‑K candidates by composite score, plus some diagnostics.

This already captures much of the **oscillatory structure** that the LTA amplitude/residual plots highlight, but it is:

- **Metric-light** on direct residual/error behaviour.
- **Brute-force** in its use of the parameter grid (no adaptive refinement, no early stopping).

---

## 2. Target Behaviour – How “Best Fit” Should Behave

Based on the proprietary tool and the sample data description:

- A **good fit** is characterised by:
  - small, unstructured residuals in the amplitude (`*_GOP`) chart after removing a smooth baseline,
  - correct **fringe spacing** and **phase** (oscillation frequency and alignment),
  - a physically plausible parameter combination (lipid/aqueous/roughness) consistent with the tear film model.
- A **bad fit** typically shows:
  - large, structured residuals (e.g., clearly misaligned oscillations),
  - incorrect or missing fringes,
  - instability across time / blinks when considered in context.

For the grid search (single-spectrum focus), we translate this into:

- **Primary objective**: minimise residual amplitude between measured and theoretical spectra after detrending / normalisation.
- **Secondary objectives**: match peak structure and phase, so that visually overlaid spectra and amplitude charts look “right”.
- **Quality awareness**: down-weight or flag fits that look numerically good but violate simple quality heuristics (e.g., out-of-range parameters, poor SNR, too few peaks).

---

## 3. Proposed Metric Stack (What to Score)

We keep the existing structural metrics and add a residual-based term plus simple quality heuristics. All metrics are normalised to \([0, 1]\) and combined via configurable weights.

### 3.1. Residual / Amplitude Match (New)

**Intent**: approximate what the LTA amplitude (`*_GOP`) chart expresses visually.

- **Inputs**:
  - `measurement.detrended` (from `PreparedMeasurement`),
  - `theoretical.detrended` (from `PreparedTheoreticalSpectrum`),
  - optionally also the raw reflectance for cross-checks.
- **Computation**:
  - Compute residual \(r = m - t\) on the detrended domain.
  - Derive standard fit metrics:
    - RMSE, MAE, and \(R^2\) as already available in `calculate_fit_metrics`.
  - Map residual error to a score, e.g.:
    - `residual_score = exp(-rmse / tau_rmse)` with configurable `tau_rmse`,
    - or use a bounded transform of \(R^2\) if it behaves well across cases.
- **Implementation sketch**:
  - Extend `src/analysis/metrics.py` with `residual_score(...)` that:
    - uses `calculate_fit_metrics` internally,
    - returns a `MetricResult` with `score` and diagnostics (`rmse`, `mae`, `r2`, etc.).
  - Add a `residual` block to `analysis.metrics` in `config.yaml`:
    - `tau_rmse`, allowed maximum RMSE, and any clipping thresholds.

This metric should carry **significant weight** because clinicians visually focus on residual/amplitude plots.

### 3.2. Peak-Based Metrics (Keep, Possibly Reweight)

Retain the existing peak metrics with minor adjustments to naming and configuration:

- `peak_count_score`:
  - Already implemented; leave behaviour as-is but ensure config is driven by `config.yaml` via `analysis.metrics.peak_count`.
- `peak_delta_score`:
  - Already implemented; ensure it uses names consistent with config (`tolerance_nm`, `tau_nm`, `penalty_unpaired`).
- **Config**:
  - Move any legacy `peak_matching` config into `analysis.metrics.peak_count` and `analysis.metrics.peak_delta`.

These metrics check that the **number and positions of fringes** are realistic and aligned, complementing the residual metric.

### 3.3. Phase / Frequency-Overlap Metric (Keep)

`phase_overlap_score` is conceptually well aligned with the LTA’s concern about oscillatory structure:

- Keep as implemented, with configuration under `analysis.metrics.phase_overlap`:
  - `resample_points`,
  - FFT window type.
- Consider allowing a separate weight for phase vs magnitude in future, but that can remain out-of-scope for now.

### 3.4. Simple Quality Heuristics (Optional, Lightweight)

Based on spectral quality discussions (SNR, baseline stability, etc.), we can add **optional, cheap quality terms**:

- **Signal quality / SNR heuristic**:
  - Inspect variance of the detrended measurement vs raw noise floor.
  - Flag spectra with too low variance or extreme noise as low-confidence fits.
- **Coverage check**:
  - Ensure measurement wavelength range sufficiently overlaps the theoretical wavelength range; penalise if coverage is poor.

Implementation pattern:

- Implement a `quality_score` function that:
  - derives one or more simple flags/scores in \([0, 1]\),
  - is cheap to compute and reused for all candidates (i.e., mostly measurement-only).
- Incorporate `quality_score` either as:
  - a multiplicative term on the composite (`composite *= quality_score`), or
  - a separate score with a low weight in the composite.

### 3.5. Composite Score Definition

Update the composite score to explicitly include the new residual term (and optionally quality):

- **Component scores**:
  - `residual`,
  - `peak_count`,
  - `peak_delta`,
  - `phase_overlap`,
  - optional `quality`.
- **Config** (`config.yaml`):
  - Under `analysis.metrics.composite.weights`, define weights for each component:
    - `residual`, `peak_count`, `peak_delta`, `phase_overlap`, `quality`.
- **Behaviour**:
  - `composite_score = weighted_average(component_scores, weights)`,
  - Normalise so that the sum of weights is treated as 1; fall back to equal weights if all are zero.

The **default recommendation** is to give **residual** the largest weight, then phase/peaks, then any quality term.

---

## 4. Search Strategy & Performance Plan

The current exhaustive grid scales as:

- \(N = N_{\text{lipid}} \times N_{\text{aqueous}} \times N_{\text{roughness}}\).

This can become expensive quickly, particularly because generating each theoretical spectrum is non-trivial.

We adopt a **coarse-to-fine grid strategy** with simple heuristics, not heavy optimisation frameworks, to respect the project’s preference for boring, proven solutions.

### 4.1. Stage 1 – Coarse Grid Search

**Goal**: find promising regions in parameter space quickly.

- Use a **coarser step size** in each dimension (e.g., 3–7 points per axis).
- Evaluate the composite score for all coarse combinations.
- Keep the top‑K coarse candidates (configurable, e.g. `coarse_top_k = 20`).

Config additions:

- In `config.yaml`, under a new `analysis.grid_search` section:
  - `coarse.lipid.min / max / step`,
  - `coarse.aqueous.min / max / step`,
  - `coarse.roughness.min / max / step`,
  - `coarse.top_k` (how many coarse best candidates to refine).

Implementation notes:

- Implement a helper in `run_grid_search.py` that:
  - builds a `ParameterGrid` (or equivalent) for coarse ranges,
  - scores each candidate,
  - returns the top‑K as `(lipid, aqueous, roughness, composite_score)` tuples.

### 4.2. Stage 2 – Local Refinement Grids

**Goal**: refine around the best coarse candidates with finer resolution.

For each of the top‑K coarse results:

- Construct a **local grid** around that point:
  - Narrow ranges centred on the coarse best, e.g. \(\pm \Delta_{\text{lipid}}\), \(\pm \Delta_{\text{aqueous}}\), \(\pm \Delta_{\text{roughness}}\),
  - Finer steps.
- Evaluate all combinations in each local grid.
- Keep the **global best** across all refined grids.

Config additions:

- Under `analysis.grid_search.refine`:
  - `lipid.window_nm`, `lipid.step_nm`,
  - `aqueous.window_nm`, `aqueous.step_nm`,
  - `roughness.window_A`, `roughness.step_A`,
  - `max_refine_candidates` (optional cap on total refined evaluations).

Implementation notes:

- Implement a small helper that, given a coarse best candidate and refine config, yields a local parameter grid (clamping to global min/max).
- Use the same scoring pipeline; no need for new metric code.

### 4.3. Early-Stopping Heuristics (Optional)

To reduce evaluation time further, add simple early-stopping options:

- **Max evaluations**:
  - CLI flag `--max-spectra` already exists; wire it into both coarse and refine stages.
- **Score threshold**:
  - Optional `--target-score` or config field:
    - If any candidate reaches `composite >= target_score`, allow early termination of the search.

Implementation notes:

- In the coarse and refine loops, check `max_spectra` and `target_score` at each iteration and break early if conditions are met.

### 4.4. Parallelism & Caching (Implementation-Level Guidance)

We do not prescribe a specific parallel framework here, but we should:

- **Encourage**:
  - Batching evaluation of parameter combinations when the underlying DLL or generator makes vectorised evaluation possible.
  - Optional use of Python’s `multiprocessing` or process pools for independent spectra when/if needed.
- **Reuse**:
  - Continue to support precomputed `grid.npy` files for large sweeps where the cost of generation dominates.
- **Cache**:
  - Avoid recomputing measurement features (already done),
  - Consider caching intermediate theoretical features if the same parameter combinations are revisited.

These are implementation notes rather than requirements and can be phased in as needed.

---

## 5. Recommended Parameter Counts & Practical Limits

As a guideline, assuming a moderate CPU and current generator performance:

- **Coarse grid**:
  - Aim for **\(3–7\) points per axis**, so total combinations:
    - \(N_{\text{coarse}} \approx 27\) (3³) up to \(343\) (7³),
  - This should complete quickly (well under a second to a few seconds).
- **Refined grids**:
  - For each coarse best, target **\(5–9\) points per axis**:
    - \(N_{\text{refine-per-region}} \approx 125\) (5³) to \(729\) (9³),
  - With `coarse_top_k` around 10 and a cap on total refined evaluations, this keeps total evaluations in the tens of thousands.
- **Hard caps**:
  - Use `--max-spectra` and/or `analysis.grid_search.max_total_evals` to guard against accidentally huge grids.

These values should be tuned empirically using the provided `good_fit`/`bad_fit` samples to balance runtime against reliability of finding the clinically accepted best-fit region.

---

## 6. Required Code Changes (Summary)

This section summarises concrete work items to implement the plan.

### 6.1. Metrics & Scoring

- **Add residual metric**:
  - New function `residual_score(...)` in `src/analysis/metrics.py`:
    - Uses `measurement.detrended` and `theoretical.detrended`,
    - Calls `calculate_fit_metrics` to get RMSE/MAE/R²,
    - Returns a `MetricResult` with a [0, 1] score plus diagnostics.
- **Wire residual into composite**:
  - Extend `score_spectrum(...)` to compute `residual_score` and include it in:
    - `component_scores`,
    - `diagnostics`,
    - `composite_score` via weights from config.
- **Config updates**:
  - Add `analysis.metrics.residual` and extend `analysis.metrics.composite.weights` in `config.yaml`.
  - Align names used in code and config (e.g., `peak_count`, `peak_delta`, `phase_overlap`, `residual`).

### 6.2. Grid Search Orchestration

- **Introduce coarse + refine workflow** in `run_grid_search.py`:
  - Add helpers:
    - `_build_parameter_grid(config, mode="coarse"|"refine", center=None)` to construct axes,
    - `_evaluate_grid(iterator, measurement_features, analysis_cfg, metrics_cfg, max_spectra, target_score)` to return top‑K.
  - Add CLI options:
    - `--search-mode` with values `full` (current exhaustive) and `coarse-fine` (new default),
    - `--target-score` to allow early stopping when a good enough fit is found.
  - Implement coarse stage → refine around top‑K → final best candidate selection.

### 6.3. Configuration & Documentation

- **Extend `config.yaml`**:
  - Add `analysis.grid_search.coarse` and `.refine` sections as described above.
  - Add `analysis.metrics.residual` and update `.composite.weights`.
- **Docs**:
  - Keep `docs/grid_search_plan.md` as a high-level overview.
  - Treat this `docs/grid_search_plan_v2.md` as the authoritative implementation guide for:
    - metric behaviour,
    - search strategy,
    - configuration layout.

### 6.4. Testing & Validation

- **Unit tests**:
  - Synthetic spectra with known residuals and peak structures to validate:
    - `residual_score`,
    - composite score behaviour as weights are varied.
- **Integration tests**:
  - Tiny grids (e.g., 2×2×2) where the analytically known best candidate should be selected by:
    - full exhaustive search,
    - coarse‑to‑fine search (they should agree).
- **Empirical validation with sample data**:
  - For several cases in `data/sample_data/good_fit`:
    - Compare grid-search best candidates vs proprietary `*_BestFit.txt` choices.
  - Ensure the new composite score ranks clinically acceptable fits at the top and clearly separates typical `good_fit` from `bad_fit` examples.

---

## 7. Open Questions / To Refine with Domain Experts

- How should residual vs peak vs phase contributions be weighted across different regimes (e.g., low SNR, few peaks, extreme aqueous thickness)?
- Are there additional proprietary quality heuristics from `Spectra_Quality_Components_Analysis.md` that should be explicitly codified (e.g., rules about minimum fringe count, specific artefacts to avoid)?
- Do clinicians prefer slightly smoother spectra that trade a tiny residual increase for better interpretability, and if so, should we encode that preference in the composite score?

These questions should be revisited once more field data and feedback from clinicians are available.

