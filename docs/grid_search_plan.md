# Grid Search Plan for Spectrum Matching

## Objectives
- **Goal:** rank each (lipid, aqueous, roughness) tuple by how well its theoretical spectrum matches a measured spectrum using three metrics: (1) peak-count agreement, (2) delta between paired peaks, and (3) a phase-overlap / frequency-domain similarity term.
- **Deliverables:** reusable scoring utilities, a grid-search driver (CLI + optional Streamlit hooks), configurable metric weights/tolerances, and reporting artifacts (top-N table plus serialized results).

## Existing Building Blocks
- **Parameter generation:** `make_single_spectrum_calculator` and `generate_grid` already create spectra over parameter grids (`src/tear_film_generator.py:307-415`). We can re-use the single-spectrum calculator for on-demand evaluations or reuse saved `grid.npy` files for batch scoring.
- **Measurement ingestion & preprocessing:** The Streamlit helpers provide detrending (`src/streamlit_app.py:100-127`), peak detection (`src/streamlit_app.py:130-143`), and measurement loading/interpolation utilities (`src/streamlit_app.py:146-209`). These functions can be moved into a shared module so both CLI and UI share identical logic.
- **Configuration knobs:** `config.yaml` exposes measurement/analysis defaults (file locations, detrending cutoff, peak prominence) at `config.yaml:37-88`. Extending this file keeps new tolerances/versioning user-editable without code changes.

## Proposed Workflow
1. **Select measurement + parameter subspace**
   - User specifies a measured spectrum (file path or glob) plus which preset (or explicit min/max ranges) to evaluate.
   - Load analysis config (tolerances, weights, FFT options) from a new `analysis.metrics` section.
2. **Preprocess the measurement**
   - Load via existing TXT reader, drop NaNs, optionally smooth.
   - Detrend with Butterworth high-pass; create both raw and detrended vectors for later metrics.
   - Cache peak table (λ, amplitude, prominence) and FFT/phase representation for re-use.
3. **Enumerate parameter combinations**
   - Option A (preferred for small sweeps): reuse saved `grid.npy` computed once by `run_tear_film_generator.py`.
   - Option B (for ad-hoc scans): call `single_spectrum(lipid, aqueous, rough)` directly inside a `ParameterGrid` iterator (from `sklearn.model_selection`) or via `itertools.product` when scikit-learn is unavailable.
4. **Evaluate metrics per spectrum**
   - Interpolate theoretical spectrum to measurement wavelengths if needed.
   - Compute peak alignment, peak delta, and phase-overlap scores (details below).
   - Normalize/weight each metric, producing a composite fitness value.
5. **Aggregate & report**
   - Keep streaming top-K queue to avoid storing every spectrum when sweeps are huge.
   - Persist detailed CSV/JSON (parameters, metrics, intermediate counts) and optionally plot overlays for best candidates.
6. **Integrate with Streamlit**
   - Surface “Search best fit” button that triggers the same backend search module, displays ranked table, and lets the user jump sliders to inspect any top candidate.

## Metric Definitions & Implementation Plan

### 1. Peak Count Agreement
- **Inputs:** detrended measured spectrum, detrended theoretical spectrum, shared prominence/height thresholds.
- **Steps:**
  1. Detect peaks via `detect_peaks` on both spectra (re-using function from `src/streamlit_app.py:130-135`).
  2. Apply wavelength tolerance (e.g., ±5 nm) to allow slight misalignments.
  3. Score = `1 - (|N_meas - N_theor_matched| / max(N_meas, 1))`, where `N_theor_matched` counts theoretical peaks that have a measured counterpart within tolerance.
  4. Store unmatched peaks for debugging.
- **Config knobs:** tolerance window, prominence override, detrending filter order (all extend `config.yaml` under `analysis.peak_matching`).

### 2. Peak Delta Metric
- **Inputs:** ordered peak wavelength lists from both spectra.
- **Steps:**
  1. Use greedy nearest-neighbor pairing (or Hungarian algorithm if counts diverge) to map theoretical peaks to measured peaks within tolerance.
  2. Compute wavelength deltas (`|λ_theor - λ_meas|`) and optional amplitude deltas.
  3. Aggregate into RMS or percentile score; e.g., `exp(-mean_delta / tau)` to keep score ∈ (0,1].
  4. Penalize unpaired peaks (add fixed cost per unmatched peak).
- **Data reuse:** share detected-peak tables from metric #1 to avoid duplicate work.

### 3. Phase Overlap / Frequency Similarity
- **Goal:** quantify how similar the oscillatory structure is, independent of absolute intensity.
- **Steps:**
  1. Interpolate both spectra onto a uniform wavelength grid (already available from theoretical wavelengths, see `prepare_wavelengths_from_config` in `src/tear_film_generator.py:296-305`).
  2. Apply windowing (e.g., Hann) to reduce edge effects.
  3. Compute FFTs, derive magnitude + phase arrays.
  4. Normalize magnitudes (unit energy) to avoid bias, then compute complex correlation `C = Σ (T(ω) * conj(M(ω)))`.
  5. Phase overlap score = `|C| / (||T|| * ||M||)` or separate magnitude/phase penalties (e.g., `cos(Δφ)` averaged over dominant frequency bins).
  6. Optionally complement with time-domain cross-correlation to account for residual shifts.
- **Outputs:** scalar score plus diagnostic data (dominant frequencies, phase offsets).

### Composite Scoring
- Normalize each metric to 0‑1.
- Combine via configurable weights `w_peak_count + w_peak_delta + w_phase = 1`.
- Provide hooks to add future metrics (RMSE, derivative alignment, etc.).

## Implementation Steps
1. **Shared analysis module**
   - Extract `load_measurement_files`, `detrend_signal`, `detect_peaks`, etc., into `src/analysis/measurement_utils.py`.
   - Refactor Streamlit imports to use the shared module to keep a single source of truth.
2. **Configuration updates**
   - Extend `config.yaml` with `analysis.metrics` block (weights, tolerances, FFT settings).
   - Add CLI flags for overrides (`--peak-tol-nm`, `--metric-weights 0.4 0.4 0.2`, etc.).
3. **Metric implementations**
   - Create `src/analysis/metrics.py` with pure functions: `peak_count_score(...)`, `peak_delta_score(...)`, `phase_overlap_score(...)`.
   - Include caching helpers so each theoretical spectrum computes peaks/FFTs at most once per evaluation.
4. **Grid-search orchestrator**
   - New script `run_grid_search.py`:
     - Loads config & measurement.
     - Builds iterator over parameter tuples (using `sklearn.model_selection.ParameterGrid` when available; fallback to `itertools.product`).
     - For each combo, generates/loads spectrum, evaluates metrics, and maintains top-K list.
     - Saves `results.csv`, `best_fit.json`, and optional Plotly overlays.
   - Support both on-the-fly computation and reusing existing `outputs/spectra_*/grid.npy`.
5. **Streamlit integration**
   - Add “Grid Search” tab/button that triggers the orchestrator (could run synchronously for small sweeps or spawn background thread).
   - Display ranked table, allow selecting a row to update sliders + overlay measurement/theoretical curves (`src/streamlit_app.py` tabs at lines ~240 onward).
6. **Testing & validation**
   - Unit tests for each metric (synthetic spectra with known peaks/phase).
   - Integration test that runs a tiny grid (e.g., 2×2×2) against a mock measurement and asserts the top candidate.
   - Regression test ensuring CLI + Streamlit use identical metric outputs.

## Additional Considerations
- **Performance:** Reuse interpolated measurement arrays and FFTs; optionally cache theoretical FFTs when iterating over large grids.
- **Data quality:** Provide warnings when measurement wavelength range does not fully cover theoretical range; enforce re-sampling rules.
- **Extensibility:** Keep metrics modular so future researchers can plug in alternative similarity measures (cross-correlation, derivative peaks, etc.).
- **Documentation:** Update README and Streamlit help text to explain the new metrics and how to interpret scores; include visualization of peak matching for best candidates.
