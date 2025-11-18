## Grid Search Plan v2 – Spectra Best-Fit & Efficiency for Interactive Workflows

This document refines the original `grid_search_plan.md` using insights from the proprietary LTA software's interpretation history (as captured in `Spectra_Quality_Components_Analysis.md`) and general best practices for spectral fitting and grid search efficiency.

The goals are:

- **Metric alignment**: make the composite score reflect how the LTA tool visually judges "good" vs "bad" fits (amplitude/residual behaviour, fringe structure, and stability).
- **Search efficiency**: minimise the number of evaluated spectra while still reliably finding the same best-fit region the operator would pick.
- **Interactive performance**: enable real-time exploration via sliders and fast "suggest best-fit" searches to reduce operator time from hours to minutes.
- **Temporal smoothness**: when processing sequential spectra, minimize parameter jumps to produce clean scatter plots with minimal manual correction.

This plan describes **what** should be implemented and **where**, without forcing a particular low-level optimisation strategy.

**Target use case**: An interactive application where operators (e.g., Shlomo) adjust thickness parameters via sliders, visualize theoretical vs measured waveforms in real-time, and trigger automated grid searches to quickly find optimal fits. The system augments human judgment rather than replacing it.

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

Based on spectral quality discussions (SNR, baseline stability, etc.), we can add **optional, cheap quality checks**:

- **Signal quality / SNR heuristic**:
  - Inspect variance of the detrended measurement vs raw noise floor.
  - Flag spectra with too low variance or extreme noise as low-confidence fits.
- **Coverage check**:
  - Ensure measurement wavelength range sufficiently overlaps the theoretical wavelength range; penalise if coverage is poor.
- **Peak count minimum**:
  - Ensure measurement has at least a minimum number of peaks (e.g., 3–5) for reliable fitting.

Implementation pattern (two options):

**Option A: Fast pre-filters (recommended for interactive mode)**:
- Implement `quick_quality_gate(measurement, config) -> bool` that:
  - Runs once per measurement (not per candidate),
  - Returns `True/False` to indicate if measurement is suitable for fitting,
  - Used to skip grid search entirely for unusable spectra.
- Add similar per-candidate gates if needed (e.g., parameter out of physical range).

**Option B: Quality score (for batch/automated mode)**:
- Implement a `quality_score(measurement, config) -> MetricResult` that:
  - Derives one or more simple flags/scores in \([0, 1]\),
  - Is cheap to compute and reused for all candidates (mostly measurement-only).
- Incorporate `quality_score` either as:
  - A multiplicative term on the composite (`composite *= quality_score`), or
  - A separate score with a low weight in the composite.

**For interactive workflows**, prefer **Option A** (fast gates) to avoid wasting time scoring bad candidates. The operator can visually judge quality and override as needed.

### 3.5. Temporal Continuity Score (Optional, for Sequential Processing)

**Intent**: Penalize large jumps between adjacent timepoints to produce smooth scatter plots and help operators avoid discontinuities.

- **Inputs**:
  - Current candidate parameters `(lipid, aqueous, roughness)`,
  - Previous timepoint's best-fit parameters (if available).
- **Computation**:
  - Calculate parameter-wise jumps:
    - `lipid_jump = |lipid_current - lipid_previous|`,
    - `aqueous_jump = |aqueous_current - aqueous_previous|`,
    - `roughness_jump = |roughness_current - roughness_previous|`.
  - Apply exponential penalty:
    - `continuity_score = exp(-(lipid_jump/tau_lipid + aqueous_jump/tau_aqueous + roughness_jump/tau_roughness))`.
- **Implementation sketch**:
  - Add `temporal_continuity_score(...)` to `src/analysis/metrics.py`:
    - accepts current and previous parameter dicts,
    - returns `MetricResult` with score and diagnostics.
  - Config under `analysis.metrics.temporal_continuity`:
    - `tau_lipid_nm`, `tau_aqueous_nm`, `tau_roughness_A` (tolerance parameters),
    - `enabled: true/false`.
- **Usage**:
  - Enable when processing time-series data (e.g., sequential spectra from a measurement stack).
  - Include in composite score with low-to-medium weight.
  - For single-spectrum analysis, disable or set weight to zero.

This metric directly addresses the goal of minimizing "jumps" in thickness scatter plots that operators currently spend significant time manually correcting.

### 3.6. Composite Score Definition

Update the composite score to explicitly include the new residual term (and optional metrics):

- **Component scores**:
  - `residual`,
  - `peak_count`,
  - `peak_delta`,
  - `phase_overlap`,
  - optional `quality`,
  - optional `temporal_continuity` (when processing sequential timepoints).
- **Config** (`config.yaml`):
  - Under `analysis.metrics.composite.weights`, define weights for each component:
    - `residual`, `peak_count`, `peak_delta`, `phase_overlap`, `quality`, `temporal_continuity`.
- **Behaviour**:
  - `composite_score = weighted_average(component_scores, weights)`,
  - Normalise so that the sum of weights is treated as 1; fall back to equal weights if all are zero.

The **default recommendation** is to give **residual** the largest weight, then phase/peaks, then any quality/continuity terms.

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

### 4.4. Context-Aware Search Initialization (for Interactive/Sequential Workflows)

**Goal**: Use prior information to narrow search space and accelerate convergence.

When processing sequential timepoints or working in an interactive slider-based UI:

- **Seed coarse grid** around previous timepoint's best-fit:
  - Instead of using global parameter ranges, centre the coarse grid around the previous result.
  - Example:
    - Global range: lipid 20–120 nm,
    - Previous best: lipid 45 nm,
    - Coarse range: lipid 35–55 nm (±10 nm window).
- **Adaptive range adjustment**:
  - If previous fit had **high confidence** (composite score > threshold):
    - Use narrow window (faster search).
  - If previous fit had **low confidence** or after blink/event:
    - Revert to wider window or global range.
- **Fallback**:
  - Always allow operator override to force global search if temporal seeding produces poor results.

Config additions:

- Under `analysis.grid_search`:
  - `use_temporal_seeding: true/false`,
  - `temporal_window_multiplier: 0.3` (fraction of global range to search around previous best),
  - `confidence_threshold_for_narrow_window: 0.8` (composite score threshold).

Implementation notes:

- In `run_grid_search.py`, add optional parameter `previous_best_params` to search functions.
- When present and `use_temporal_seeding` is enabled:
  - Compute coarse grid ranges as `previous_value ± (global_range * window_multiplier)`.
  - Clamp to global min/max bounds.

This optimization is particularly valuable for the interactive workflow where operators process stacks of sequential spectra and expect smooth parameter evolution.

### 4.5. Parallelism & Caching (Implementation-Level Guidance)

We do not prescribe a specific parallel framework here, but we should:

- **Encourage**:
  - Batching evaluation of parameter combinations when the underlying DLL or generator makes vectorised evaluation possible.
  - Optional use of Python's `multiprocessing` or process pools for independent spectra when/if needed.
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
    - `_build_parameter_grid(config, mode="coarse"|"refine", center=None, previous_best=None)` to construct axes,
    - `_evaluate_grid(iterator, measurement_features, analysis_cfg, metrics_cfg, max_spectra, target_score, previous_best_params=None)` to return top‑K.
  - Add CLI options:
    - `--search-mode` with values `full` (current exhaustive) and `coarse-fine` (new default),
    - `--target-score` to allow early stopping when a good enough fit is found,
    - `--previous-best-params` to enable temporal seeding from previous timepoint,
    - `--mode` with values `single` (default) and `batch` (for processing stacks).
  - Implement coarse stage → refine around top‑K → final best candidate selection.
  - When `previous_best_params` provided and temporal seeding enabled:
    - Center coarse grid around previous best instead of global ranges.
    - Include temporal continuity score in composite (if configured).

### 6.3. Configuration & Documentation

- **Extend `config.yaml`**:
  - Add `analysis.grid_search.coarse` and `.refine` sections as described in Section 4.
  - Add `analysis.grid_search.use_temporal_seeding`, `.temporal_window_multiplier`, and `.confidence_threshold_for_narrow_window` as described in Section 4.4.
  - Add `analysis.metrics.residual` with `tau_rmse` and other residual-specific settings.
  - Add `analysis.metrics.temporal_continuity` with `tau_lipid_nm`, `tau_aqueous_nm`, `tau_roughness_A`, and `enabled` flag.
  - Update `analysis.metrics.composite.weights` to include weights for `residual`, `peak_count`, `peak_delta`, `phase_overlap`, `quality` (optional), and `temporal_continuity` (optional).
  - Add `analysis.quality_gates` section (if using Option A from Section 3.4) with thresholds for:
    - `min_peaks`,
    - `min_signal_amplitude`,
    - `min_wavelength_coverage`.
- **Docs**:
  - Keep `docs/grid_search_plan.md` as a high-level overview.
  - Treat this `docs/grid_search_plan_v2.md` as the authoritative implementation guide for:
    - metric behaviour,
    - search strategy,
    - configuration layout,
    - interactive workflow considerations.

### 6.4. Testing & Validation

- **Unit tests**:
  - Synthetic spectra with known residuals and peak structures to validate:
    - `residual_score`,
    - `temporal_continuity_score`,
    - composite score behaviour as weights are varied.
- **Integration tests**:
  - Tiny grids (e.g., 2×2×2) where the analytically known best candidate should be selected by:
    - full exhaustive search,
    - coarse‑to‑fine search (they should agree).
  - Sequential timepoint tests:
    - Verify temporal seeding narrows search appropriately,
    - Verify continuity score reduces parameter jumps.
- **Empirical validation with sample data**:
  - For several cases in `data/sample_data/good_fit`:
    - Compare grid-search best candidates vs proprietary `*_BestFit.txt` choices.
  - Ensure the new composite score ranks clinically acceptable fits at the top and clearly separates typical `good_fit` from `bad_fit` examples.
  - For time-series stacks:
    - Verify scatter plots are smoother with temporal continuity enabled vs disabled.

### 6.5. Interactive Mode & UI Considerations

**Context**: This grid search will power an interactive application where operators (e.g., Shlomo) adjust sliders and visualize fits in real-time, replacing hours of manual work with minutes of guided exploration.

#### 6.5.1. Real-Time Evaluation Mode

For live slider-based exploration:

- **Single-candidate evaluation**:
  - When sliders move, evaluate current parameter combination immediately (no grid search).
  - Display component scores separately in UI:
    - Residual score (with RMSE/MAE/R² diagnostics),
    - Peak match score,
    - Phase overlap score,
    - Overall composite score.
  - Update theoretical waveform overlay in real-time.

- **Implementation**:
  - Add `evaluate_single_candidate(params, measurement_features, config)` function.
  - Callable from UI layer with sub-100ms latency target.
  - DLL call should be asynchronous to avoid blocking UI.

#### 6.5.2. "Suggest Best-Fit" Workflow

Operator-triggered automated search:

- **Trigger**: Button click or keyboard shortcut in UI.
- **Behaviour**:
  - Use current slider values as seed (context-aware search).
  - Run coarse-to-fine grid search in background thread.
  - Present top-K candidates (e.g., top 5) ranked by composite score.
  - Allow operator to:
    - Accept suggestion (update sliders to best-fit),
    - Review alternatives (click through top-K),
    - Reject and continue manual exploration.

- **Implementation**:
  - CLI `run_grid_search.py` functionality wrapped in Python API.
  - Return `List[CandidateResult]` with parameters, scores, and diagnostics.
  - UI renders candidate list with preview thumbnails.

#### 6.5.3. Confidence Indicators

Visual cues to help operator decide when to trust automated suggestions:

- **Confidence levels**:
  - **High** (composite score > 0.85): Green indicator, "Excellent fit – safe to accept".
  - **Medium** (0.65–0.85): Yellow indicator, "Good fit – review recommended".
  - **Low** (< 0.65): Red indicator, "Poor fit – manual adjustment needed".

- **Additional diagnostics**:
  - Show residual RMSE value alongside score.
  - Flag large jumps from previous timepoint (if temporal continuity enabled).
  - Highlight when peak counts differ significantly.

- **Implementation**:
  - Add `confidence_level(composite_score, config)` helper function.
  - UI displays color-coded badge and tooltip with diagnostics.

#### 6.5.4. Sequential Processing Mode

For processing entire measurement stacks:

- **Workflow**:
  - Operator selects time range or entire stack.
  - System processes timepoints sequentially:
    - Uses previous best-fit to seed each search (temporal seeding),
    - Applies temporal continuity scoring to prefer smooth evolution.
  - Generates preliminary scatter plot.
  - Operator reviews and manually corrects outliers/jumps.

- **Implementation**:
  - Add batch processing mode to `run_grid_search.py`:
    - `--mode batch --input-stack path/to/stack --output-results path/to/csv`.
  - Enable temporal features by default in batch mode.
  - Save intermediate results so operator can resume/modify.

#### 6.5.5. Performance Targets

To achieve the goal of reducing operator time from hours to minutes:

- **Single-candidate evaluation**: < 100ms (real-time slider feedback).
- **Coarse grid search**: < 2 seconds (quick suggestions).
- **Full coarse-to-fine search**: < 10 seconds (high-quality best-fit).
- **Batch processing (100 timepoints)**: < 10 minutes (with temporal seeding).

These targets assume moderate hardware (modern CPU, no GPU required) and efficient DLL calls.

---

## 7. Open Questions / To Refine with Domain Experts

- How should residual vs peak vs phase contributions be weighted across different regimes (e.g., low SNR, few peaks, extreme aqueous thickness)?
- What is the optimal balance between temporal continuity (smooth scatter plots) and fitting accuracy when processing sequential timepoints?
- Should confidence thresholds be adjusted based on measurement quality (e.g., lower acceptance threshold for noisy spectra)?
- What are the appropriate tau values for temporal continuity scoring (how much jump is acceptable before penalizing)?
- Do clinicians prefer slightly smoother spectra that trade a tiny residual increase for better interpretability, and if so, should we encode that preference in the composite score?

These questions should be revisited once more field data and feedback from operators (e.g., Shlomo) are available during interactive workflow testing.


