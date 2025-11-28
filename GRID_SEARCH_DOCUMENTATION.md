# Grid Search System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Initial Parameters (Yash's Baseline - H0)](#initial-parameters-yashs-baseline---h0)
3. [RGT Grid Search Implementation](#rgt-grid-search-implementation)
4. [Grid Search Strategies](#grid-search-strategies)
5. [Scoring Metrics](#scoring-metrics)
6. [Composite Score Calculation](#composite-score-calculation)
7. [Edge Case Detection and Flagging](#edge-case-detection-and-flagging)
8. [Best Score Selection](#best-score-selection)
9. [Configuration](#configuration)
10. [Results Documentation Template](#results-documentation-template)
11. [Performance Considerations](#performance-considerations)
12. [Known Limitations and Future Work](#known-limitations-and-future-work)

---

## Overview

This document describes the grid search system implemented by **RGT** for the AdOM tear film analysis application. The system builds upon the initial web application developed by **Yash**, which established the baseline parameter ranges and theoretical spectrum generation framework. RGT's implementation adds intelligent grid search algorithms, multi-metric scoring, and edge case detection to systematically find optimal tear film parameters.

**System Architecture:**
- **Yash (Initial Developer)**: Developed the web application, established baseline parameter ranges (H0), and implemented theoretical spectrum generation using the SpAnalizer DLL
- **RGT (Current Implementation)**: Implemented grid search algorithms, scoring methodology, edge case detection, and validation flagging

**Key Parameters:**
- **Lipid thickness (nm)**: Range 0-500nm (expanded beyond algorithm constraint 120-200nm to allow exploration)
- **Aqueous thickness (nm)**: Range -20-12000nm (expanded beyond algorithm constraint 680-1200nm, includes negative values)
- **Mucus roughness (Å)**: Range 0-3000Å (expanded beyond algorithm constraint 600-2600Å to allow exploration)

**Note**: While the algorithm XML defines constraints (120-200nm, 680-1200nm, 600-2600Å), the DLL accepts values outside these ranges. Client testing revealed that optimal values (e.g., lipid=60.1nm, aqueous=-2.0nm) can fall outside the algorithm constraints, so RGT's implementation uses expanded ranges for both grid search and manual slider adjustment.

**Total Search Space**: With expanded ranges and default step sizes (lipid: 5nm, aqueous: 10nm, roughness: 50Å), this creates a much larger search space. The coarse-to-fine grid search strategy efficiently explores this space.

---

## Initial Parameters (Yash's Baseline - H0)

The initial parameter ranges and constraints were established by **Yash** as the baseline hypothesis (H0) for the tear film analysis system. These ranges are based on:

1. **Algorithm Constraints**: Defined in `Algorithm_eye_test_4.xml` - the DLL's valid input ranges
2. **Biological Feasibility**: Based on known tear film layer thickness ranges
3. **Initial Testing**: Yash's preliminary testing to establish reasonable bounds

### Parameter Ranges (H0 - Yash's Original)

**Lipid Layer Thickness:**
- **Original Range**: 120-200nm (algorithm constraint from DLL)
- **Step Size**: 5nm
- **Rationale**: Algorithm constraint from DLL; typical tear film lipid layer thickness
- **Total Values**: 17 (120, 125, 130, ..., 200)

**Aqueous Layer Thickness:**
- **Original Range**: 680-1200nm (algorithm constraint from DLL)
- **Step Size**: 10nm
- **Rationale**: Algorithm constraint from DLL; typical aqueous layer thickness
- **Total Values**: 53 (680, 690, 700, ..., 1200)
- **Note**: Aqueous can be 0nm in some pathological cases, but Yash's initial range starts at 680nm

**Mucus Layer Roughness:**
- **Original Range**: 600-2600Å (algorithm constraint from DLL)
- **Step Size**: 50Å
- **Rationale**: Algorithm constraint from DLL; surface roughness range
- **Total Values**: 41 (600, 650, 700, ..., 2600)

### Expanded Parameter Ranges (RGT's Current Implementation)

**Lipid Layer Thickness:**
- **Expanded Range**: 0-500nm (beyond algorithm constraint)
- **Step Size**: 5nm (for manual sliders)
- **Rationale**: Client testing revealed optimal values (e.g., 60.1nm) outside algorithm range; DLL accepts these values
- **Total Values**: 101 (0, 5, 10, ..., 500)

**Aqueous Layer Thickness:**
- **Expanded Range**: -20-12000nm (beyond algorithm constraint, includes negative values)
- **Step Size**: 10nm (for manual sliders)
- **Rationale**: Client testing revealed optimal values (e.g., -2.0nm) outside algorithm range; negative values are valid
- **Total Values**: 1203 (-20, -10, 0, 10, ..., 12000)

**Mucus Layer Roughness:**
- **Expanded Range**: 0-3000Å (beyond algorithm constraint)
- **Step Size**: 50Å (for manual sliders)
- **Rationale**: Allows exploration beyond algorithm range; DLL accepts these values
- **Total Values**: 61 (0, 50, 100, ..., 3000)

### Fixed Parameters (Yash's Configuration)

- **Mucus Thickness**: 500nm (kept constant, not varied in grid search)

### Other Parameters Tracked by Yash's System

While not part of the grid search, Yash's original system tracked additional quality metrics:

1. **RMS (Root Mean Square) Values**:
   - `rms_minimal_ratio_from_max_in_percents`: Minimum RMS as percentage of maximum (typical: 44-50%)
   - `rms_minimal_absolute_value`: Absolute minimum RMS threshold (typical: 0.0007)
   - `wave_lenghts_range_for_spectrum_amplitude_rms`: Wavelength range for RMS calculation (typical: 750-920nm)

2. **GOP (Goodness of Peaks)**:
   - `GoodnessOfPeaksThreshold`: Maximum GOP value for acceptable spectra (typical: 75-1000, lower is better)
   - `IsCalculateGoodnessOfPeaks`: Enable/disable GOP calculation
   - `PeaksDetectionType`: Type of peak detection ('Normal', 'Small', 'None')
   - GOP is a quality metric indicating peak detection quality (lower values = better peak detection)

3. **Dif Max Abs (Difference Maximum Absolute)**:
   - `dif_max_abs_minimal_ratio_from_max_in_percents`: Minimum as percentage of maximum (typical: 50%)
   - `dif_max_abs_minimal_absolute_value`: Absolute minimum threshold (typical: 0.0001-0.000111)

**Note**: These parameters are used for quality validation in Yash's system but are not part of the grid search parameter space. RGT's implementation focuses on the three variable parameters (lipid, aqueous, roughness) while incorporating quality checks through the `measurement_quality_score` metric.

---

## RGT Grid Search Implementation

### Methodology

RGT's approach to grid search differs from exhaustive testing by using intelligent sampling and refinement strategies:

1. **Initial Sampling**: Rather than testing all 36,941 combinations, RGT uses:
   - **Random Sampling**: For quick exploration when budget is limited (<10% of search space)
   - **Coarse-to-Fine Search**: Two-stage approach that first identifies promising regions, then refines around best candidates

2. **Scoring and Ranking**: Each candidate is scored using multiple metrics:
   - Peak count matching
   - Peak wavelength alignment
   - Phase/FFT overlap
   - Residual fit quality (RMSE, MAE, R²)
   - Measurement quality validation
   - Temporal continuity (for time-series)

3. **Ruling Out Ranges**: The coarse-to-fine approach naturally excludes poor regions:
   - Coarse stage identifies top-K promising candidates
   - Refinement focuses computational resources only on these regions
   - Poor regions are implicitly excluded by not being selected for refinement

4. **Expansion Beyond Initial Ranges**: During refinement, the search expands beyond Yash's initial parameter ranges (up to acceptable/feasible limits) to discover optimal values that may lie outside the initial search space.

### Key Differences from Yash's Approach

| Aspect | Yash's System | RGT's Implementation |
|--------|--------------|---------------------|
| **Parameter Exploration** | Manual slider adjustment | Automated grid search |
| **Scoring** | Visual inspection, GOP threshold | Multi-metric composite scoring |
| **Coverage** | User-selected combinations | Systematic/random/coarse-fine strategies |
| **Validation** | GOP, RMS thresholds | Edge case detection, quality gates |
| **Parameter Ranges** | Fixed 120-200nm, 680-1200nm, 600-2600Å | Expands during refinement to acceptable ranges |

---

## Grid Search Strategies

The system supports three search strategies, each optimized for different use cases:

### 1. Coarse-to-Fine Search (Recommended - Default)

**Purpose**: Efficiently explore large parameter spaces by first identifying promising regions, then refining around the best candidates.

**How It Works:**

#### Stage 1: Coarse Search
- **Goal**: Quickly identify promising regions in parameter space
- **Method**: Uses wider step sizes to evaluate a sparse grid across the entire parameter space
- **Current Configuration** (optimized for ~20 min runtime):
  - Lipid: 0-500nm, step=40nm → **~13 values** (0, 40, 80, 120, ..., 500)
  - Aqueous: -20-12000nm, step=200nm → **~61 values** (-20, 180, 380, 580, ..., 12000)
  - Roughness: 0-3000Å, step=500Å → **7 values** (0, 500, 1000, 1500, 2000, 2500, 3000)
  - **Total coarse combinations**: ~13 × 61 × 7 = **~5,551 combinations**
  - **top_k**: 50 (increased from 20 to allow more candidates for refinement)

- **Process**:
  1. Generate theoretical spectrum for each coarse combination
  2. Score each spectrum using all metrics
  3. Sort by composite score (highest first)
  4. Select top-K candidates (current: top 50)

#### Stage 2: Refinement Search
- **Goal**: Fine-tune parameters around the best coarse candidates
- **Method**: Creates local grids centered on each top-K candidate with finer step sizes
- **Expansion**: Refinement windows can expand beyond Yash's initial parameter ranges (120-200nm, 680-1200nm, 600-2600Å) up to acceptable/feasible limits (0-500nm, 0-2000nm, 0-5000Å) to discover optimal values outside the initial search space
- **Default Configuration**:
  - For each top-K candidate, create a refinement window:
    - Lipid: ±10nm window, step=5nm → **~5 values** per candidate
    - Aqueous: ±20nm window, step=10nm → **~5 values** per candidate
    - Roughness: ±100Å window, step=25Å → **~9 values** per candidate
  - **Total refine combinations per candidate**: ~5 × 5 × 9 = **~225 combinations**
  - **Total refine combinations (50 candidates)**: ~11,250 combinations

- **Process**:
  1. For each top-K coarse candidate:
     - Define a refinement window centered on that candidate's parameters
     - Expand window beyond initial ranges if needed (clamped to acceptable ranges)
     - Generate fine-grained parameter grid within window
     - Evaluate all combinations in this local grid
  2. Combine all coarse and refined results
  3. Remove duplicates (same parameter combinations)
  4. Sort by composite score

**Advantages**:
- **Efficiency**: Evaluates ~4,680 combinations instead of 36,941 (87% reduction)
- **Quality**: Focuses computational resources on promising regions
- **Scalability**: Can handle very large parameter spaces efficiently
- **Discovery**: Can find optimal values outside initial search ranges

**When to Use**:
- Large parameter spaces (>10,000 combinations)
- Limited computational budget
- Need good results quickly
- Default recommended strategy

**Example Timeline** (current configuration):
- Coarse stage: ~5,551 combinations × 0.025s = **~2.3 minutes**
- Refine stage: ~50 candidates × ~225 combinations each = ~11,250 combinations × 0.025s = **~4.7 minutes**
- **Total: ~7-8 minutes** (optimized for ~20 min total including overhead)

**Note**: Step sizes and ranges are optimized to balance exploration coverage with runtime. The expanded parameter ranges (0-500nm lipid, -20-12000nm aqueous, 0-3000Å roughness) allow discovery of optimal values outside the original algorithm constraints.

---

### 2. Random Search

**Purpose**: Unbiased exploration when only a small subset of the search space can be evaluated.

**How It Works**:
- **Trigger Condition**: Automatically used when `max_results` is set AND covers <10% of total search space
- **Method**: 
  1. Generate all possible parameter combinations (based on Yash's ranges)
  2. Randomly sample `max_results` combinations (using fixed seed=42 for reproducibility)
  3. Evaluate only the sampled combinations
  4. Sort by composite score

**Example**:
- Total search space: 36,941 combinations (Yash's ranges)
- User sets max_results: 1,000
- Coverage: 1,000 / 36,941 = 2.7% (< 10% threshold)
- **Action**: Randomly sample 1,000 combinations from all 36,941

**Advantages**:
- **No Bias**: Avoids systematic bias toward lower parameter values
- **Reproducible**: Fixed random seed ensures same results on rerun
- **Efficient**: Only evaluates requested number of combinations

**When to Use**:
- Limited evaluation budget (<10% of search space)
- Want unbiased exploration
- Quick testing/exploration

**Limitation**:
- May miss optimal solutions if they're in unexplored regions
- Less efficient than coarse-fine for large spaces

---

### 3. Systematic (Exhaustive) Search

**Purpose**: Complete exploration of the parameter space when computational resources allow.

**How It Works**:
- **Trigger Condition**: Used when random sampling is NOT triggered (max_results ≥ 10% coverage OR max_results=None)
- **Method**:
  1. Generate parameter arrays using Yash's ranges and step sizes
  2. Nested loops: for each lipid → for each aqueous → for each roughness
  3. Evaluate each combination in order
  4. Stop when `max_results` limit reached (if set)
  5. Sort by composite score

**Example**:
- Parameter arrays: lipid=[120,125,130,...,200], aqueous=[680,690,700,...,1200], roughness=[600,650,700,...,2600]
- **Evaluation Order**: (120,680,600), (120,680,650), (120,680,700), ..., (200,1200,2600)
- If max_results=5000: Evaluates first 5,000 combinations in this order

**Advantages**:
- **Complete Coverage**: When max_results=None, evaluates entire space
- **Deterministic**: Same order every time
- **Simple**: Easy to understand and debug

**Disadvantages**:
- **Bias Risk**: If stopped early, biased toward lower parameter values
- **Slow**: Full search takes ~15-30 minutes for 36,941 combinations

**When to Use**:
- Need complete coverage
- Computational time is not a concern
- Want deterministic, reproducible results
- Full grid search button in UI

---

## Scoring Metrics

RGT's implementation uses **six independent metrics** to evaluate how well a theoretical spectrum matches a measured spectrum. Each metric returns a score between 0.0 (worst) and 1.0 (best). These metrics capture different aspects of spectral similarity that complement Yash's GOP and RMS-based quality checks.

### 1. Peak Count Score

**Purpose**: Measures how well the number of peaks matches between theoretical and measured spectra.

**Calculation**:
1. Detect peaks in both measured and theoretical spectra (using prominence threshold)
2. Match peaks within tolerance (default: 5nm wavelength difference)
3. Count matched peaks
4. Score = `1.0 - |measured_peaks - matched_peaks| / measured_peaks`
   - If measured has 5 peaks and 4 match: score = 1.0 - |5-4|/5 = **0.8**
   - If measured has 3 peaks and all 3 match: score = 1.0 - |3-3|/3 = **1.0**
   - If measured has 0 peaks and theoretical has 0: score = **1.0** (perfect match)
   - If measured has 0 peaks but theoretical has peaks: score = **0.0**

**Diagnostics**:
- `measurement_peaks`: Number of peaks in measured spectrum
- `theoretical_peaks`: Number of peaks in theoretical spectrum
- `matched_peaks`: Number of successfully matched peaks

**Configuration** (`config.yaml`):
```yaml
metrics:
  peak_count:
    wavelength_tolerance_nm: 5.0  # Maximum wavelength difference for matching
```

**Interpretation**:
- **1.0**: Perfect match in peak count
- **0.8-0.99**: Good match, minor differences
- **<0.8**: Significant mismatch in peak structure

**Relationship to Yash's System**: This metric is related to Yash's GOP (Goodness of Peaks) but focuses on peak count matching rather than peak quality assessment.

---

### 2. Peak Delta Score

**Purpose**: Measures how accurately matched peaks align in wavelength (not just that they match).

**Calculation**:
1. Match peaks between measured and theoretical (same as peak_count)
2. Calculate wavelength differences (deltas) for each matched pair
3. Compute mean delta across all matched pairs
4. Score = `exp(-mean_delta / tau_nm)` where tau_nm is a decay constant (default: 15nm)
   - If mean_delta = 0nm: score = exp(0) = **1.0** (perfect alignment)
   - If mean_delta = 15nm: score = exp(-1) = **0.368**
   - If mean_delta = 30nm: score = exp(-2) = **0.135**
5. Apply penalty for unpaired peaks: `penalty = penalty_unpaired × (unmatched_measured + unmatched_theoretical)`
6. Final score = `max(0.0, score - penalty)`

**Special Cases**:
- If both spectra have 0 peaks: score = **1.0** (perfect match)
- If no peaks match: score = **0.0**

**Diagnostics**:
- `matched_pairs`: Number of successfully matched peak pairs
- `mean_delta_nm`: Average wavelength difference between matched peaks
- `unpaired_measurement`: Number of measured peaks without matches
- `unpaired_theoretical`: Number of theoretical peaks without matches

**Configuration**:
```yaml
metrics:
  peak_delta:
    tolerance_nm: 5.0          # Maximum delta to consider a match
    tau_nm: 15.0               # Decay constant (lower = stricter)
    penalty_unpaired: 0.05      # Penalty per unpaired peak
```

**Interpretation**:
- **1.0**: Perfect peak alignment
- **0.7-0.99**: Good alignment, minor wavelength shifts
- **0.3-0.7**: Moderate alignment, some wavelength error
- **<0.3**: Poor alignment, significant wavelength mismatches

---

### 3. Phase Overlap Score

**Purpose**: Measures similarity in the frequency domain (FFT) of detrended spectra, capturing overall spectral shape.

**Calculation**:
1. Detrend both measured and theoretical spectra (remove low-frequency baseline)
2. Compute FFT (Fast Fourier Transform) of both detrended spectra
3. Calculate normalized dot product (coherence):
   - `numerator = |vdot(candidate_fft, reference_fft)|`
   - `denominator = ||candidate_fft|| × ||reference_fft||`
   - `score = numerator / denominator`
4. This is essentially the cosine similarity in frequency space

**Mathematical Formulation**:
```
score = |⟨FFT(theoretical), FFT(measured)⟩| / (||FFT(theoretical)|| × ||FFT(measured)||)
```

**Diagnostics**:
- `coherence`: Absolute value of dot product
- `norm_reference`: Norm of measured FFT
- `norm_candidate`: Norm of theoretical FFT

**Configuration**:
```yaml
metrics:
  phase_overlap:
    resample_points: 1024       # FFT resolution
    window: "hann"              # Windowing function
```

**Interpretation**:
- **1.0**: Identical frequency content (perfect match)
- **0.8-0.99**: Very similar spectral shape
- **0.5-0.8**: Moderate similarity
- **<0.5**: Different spectral characteristics

**Why It Matters**: Captures overall spectral shape and periodic patterns that peak-based metrics might miss. This complements Yash's RMS-based amplitude analysis by focusing on frequency domain characteristics.

---

### 4. Residual Score

**Purpose**: Measures direct fit quality between measured and theoretical spectra (RMSE, MAE, R²). This is similar to Yash's RMS analysis but applied to the full spectrum fit rather than amplitude RMS.

**Calculation**:
1. Align measured and theoretical spectra to same wavelength grid (interpolation)
2. **Use RAW (non-detrended) spectra** for comparison - visual quality depends on raw alignment
   - Detrended spectra remove baseline, which can hide misalignment issues
   - Raw spectra better capture visual similarity between measured and theoretical curves
3. Calculate fit metrics on raw spectra:
   - **RMSE** (Root Mean Square Error): `sqrt(mean((measured - theoretical)²))`
   - **MAE** (Mean Absolute Error): `mean(|measured - theoretical|)`
   - **R²** (Coefficient of Determination): `1 - SS_res / SS_tot`
   - **MAPE** (Mean Absolute Percentage Error): `mean(|(measured - theoretical) / measured|) × 100`
4. Convert RMSE to base score: `rmse_score = exp(-RMSE / tau_rmse)`
   - If RMSE = 0: rmse_score = exp(0) = **1.0** (perfect fit)
   - If RMSE = tau_rmse: rmse_score = exp(-1) = **0.368**
   - If RMSE = 2×tau_rmse: rmse_score = exp(-2) = **0.135**
5. Handle R²:
   - **If R² < 0**: Ignore R², use RMSE score only (negative R² doesn't necessarily mean bad visual fit)
   - **If R² ≥ 0**: Combine RMSE and R² scores: `score = 0.6 × rmse_score + 0.4 × (rmse_score × r2_score)`
     where `r2_score = max(0.3, min(1.0, 0.3 + 0.7 × R²))`
6. If `max_rmse` is set and RMSE ≥ max_rmse: score = **0.0** (hard cutoff)

**Diagnostics**:
- `rmse`: Root Mean Square Error
- `mae`: Mean Absolute Error
- `r2`: R-squared (coefficient of determination)
- `mape_pct`: Mean Absolute Percentage Error (%)

**Configuration**:
```yaml
metrics:
  residual:
    tau_rmse: 0.015            # Decay constant for RMSE (lower = stricter)
    max_rmse: 0.1               # Hard cutoff (score=0 if RMSE ≥ this)
```

**Interpretation**:
- **1.0**: Perfect fit (RMSE ≈ 0)
- **0.7-0.99**: Excellent fit (low RMSE)
- **0.3-0.7**: Good fit (moderate RMSE)
- **<0.3**: Poor fit (high RMSE)
- **0.0**: RMSE exceeds maximum threshold

**Key Improvement**: Using raw (non-detrended) spectra instead of detrended spectra for residual calculation significantly improves visual alignment. Detrending removes baseline information that's critical for visual quality assessment.

**Negative R² Handling**: When R² is negative, the system ignores it and uses only the RMSE score. This is because negative R² can occur when the theoretical spectrum has different amplitude structure but still matches well in phase/frequency domain, which is common for visually good fits.

**Relationship to Yash's System**: This metric is conceptually similar to Yash's RMS analysis but evaluates the full spectrum fit quality on raw spectra rather than amplitude RMS within a specific wavelength range.

---

### 5. Measurement Quality Score

**Purpose**: Validates that the measured spectrum is suitable for analysis (not a metric comparing spectra). This serves a similar purpose to Yash's quality gates (GOP, RMS thresholds) but uses different criteria.

**Calculation**:
Checks three quality criteria:
1. **Peak Count**: `peaks_detected ≥ min_peaks` (default: 3)
2. **Signal Amplitude**: `max(spectrum) - min(spectrum) ≥ min_signal_amplitude` (default: 0.02)
3. **Wavelength Span**: `max(wavelength) - min(wavelength) ≥ min_wavelength_span_nm` (default: 150nm)

Each check contributes to the score:
- If all checks pass: score = **1.0**
- If 2/3 pass: score = **0.67**
- If 1/3 pass: score = **0.33**
- If 0/3 pass: score = **0.0**

**Diagnostics**:
- `peak_count`: Number of detected peaks
- `signal_amplitude`: Range of signal values
- `wavelength_span_nm`: Wavelength coverage

**Configuration**:
```yaml
analysis:
  quality_gates:
    enabled: true
    min_quality_score: 0.5
    min_peaks: 3
    min_signal_amplitude: 0.02
    min_wavelength_span_nm: 150.0
```

**Interpretation**:
- **1.0**: High-quality measurement, all checks pass
- **0.5-0.99**: Acceptable quality, some checks may fail
- **<0.5**: Low-quality measurement, results may be unreliable

**Note**: This score is applied to ALL candidates for the same measurement (not per-candidate). It's a quality gate, not a comparison metric.

**Relationship to Yash's System**: This metric serves a similar purpose to Yash's GOP and RMS quality thresholds but uses different criteria (peak count, signal amplitude, wavelength span) rather than GOP values and RMS ratios.

---

### 6. Temporal Continuity Score

**Purpose**: For time-series analysis, penalizes large jumps in parameters between consecutive timepoints.

**Calculation** (only used when `temporal_continuity.enabled = true`):
1. Compare current parameters to previous timepoint's parameters
2. Calculate normalized differences:
   - `delta_lipid = |current_lipid - previous_lipid| / tau_lipid_nm`
   - `delta_aqueous = |current_aqueous - previous_aqueous| / tau_aqueous_nm`
   - `delta_roughness = |current_roughness - previous_roughness| / tau_roughness_A`
3. Score = `exp(-max(delta_lipid, delta_aqueous, delta_roughness))`
   - If no change: score = exp(0) = **1.0**
   - If change = tau: score = exp(-1) = **0.368**
   - If change = 2×tau: score = exp(-2) = **0.135**

**Diagnostics**:
- `delta_lipid_nm`: Change in lipid thickness
- `delta_aqueous_nm`: Change in aqueous thickness
- `delta_roughness_A`: Change in roughness

**Configuration**:
```yaml
metrics:
  temporal_continuity:
    enabled: false              # Set to true for time-series
    tau_lipid_nm: 10.0          # Expected change scale
    tau_aqueous_nm: 30.0
    tau_roughness_A: 150.0
```

**Interpretation**:
- **1.0**: Smooth transition from previous timepoint
- **0.5-0.99**: Moderate change, acceptable
- **<0.5**: Large jump, may indicate error

**When to Use**: Only for analyzing sequential measurements (e.g., temporal tear film evolution).

---

## Composite Score Calculation

The composite score combines all individual metric scores into a single ranking value using **weighted averaging**. This allows RGT's system to rank candidates based on multiple criteria simultaneously, providing a more comprehensive assessment than single-metric approaches.

### Formula

```
composite_score = Σ(weight_i × score_i) / Σ(weights)
```

Where:
- `weight_i` = weight for metric `i` (from config)
- `score_i` = score for metric `i` (0.0 to 1.0)
- Sum is over all enabled metrics

### Default Weights (Updated for Visual Quality)

From `config.yaml` (current implementation):
```yaml
metrics:
  composite:
    weights:
      phase_overlap: 0.7      # 70% weight - primary indicator of visual quality
      residual: 0.2           # 20% weight - RMSE/MAE matter, R² ignored when negative
      peak_count: 0.0         # Removed - not correlated with visual quality
      peak_delta: 0.0         # Removed - not correlated with visual quality
      quality: 0.1            # 10% weight (if enabled)
      temporal_continuity: 0.1 # 10% weight (if enabled)
```

**Rationale**: Analysis of client's best-fit values revealed that visual quality correlates strongly with phase_overlap and overall error (RMSE/MAE), but NOT with peak matching. Client's values had excellent phase_overlap (0.88) and good visual fit despite poor peak matching (0-1 matches out of 8-9 peaks). Therefore, peak metrics were removed and phase_overlap was made dominant.

**Note**: Weights don't need to sum to 1.0 - they're normalized automatically.

### Example Calculation (Current Weights)

Assume a candidate has these scores:
- `phase_overlap_score = 0.88` (excellent phase alignment)
- `residual_score = 0.66` (good RMSE, R² = 0.69)
- `quality_score = 1.0` (measurement quality check passed)
- `temporal_continuity_score = 0.95` (smooth transition, if enabled)

**Calculation** (peak_count and peak_delta weights are 0.0, so excluded):
```
numerator = (0.7 × 0.88) + (0.2 × 0.66) + (0.1 × 1.0) + (0.1 × 0.95)
         = 0.616 + 0.132 + 0.10 + 0.095
         = 0.943

denominator = 0.7 + 0.2 + 0.1 + 0.1
            = 1.1

composite_score = 0.943 / 1.1 = 0.857
```

**Note**: Peak metrics (peak_count, peak_delta) are excluded from calculation since their weights are 0.0.

### Weight Normalization

If all weights are zero or negative, the system falls back to **equal weights**:
```
composite_score = mean(all_scores)
```

### Interpretation

- **0.9-1.0**: Excellent match across all metrics
- **0.7-0.9**: Good match, minor issues in some metrics
- **0.5-0.7**: Moderate match, some significant discrepancies
- **<0.5**: Poor match, major issues

---

## Edge Case Detection and Flagging

RGT has implemented a comprehensive edge case detection and flagging system to identify results that require special attention. This addresses the requirement for "validation methodology or flagging to the results" mentioned in the project requirements.

### What Gets Flagged

The system flags candidates that fall into the following categories:

1. **Parameters Outside Client Accepted Ranges**: Values outside the client's business-defined acceptable limits
   - Example: `lipid_outside_client_range(600nm, client_range: 9-250nm)`
   - Client accepted ranges: Lipid 9-250nm, Aqueous 800-12000nm, Roughness 600-2750Å
   - Search space is expanded beyond these ranges to allow discovery of optimal values

2. **Exceptionally High Scores**: Composite score ≥ 0.9
   - Flagged as `exceptional_score`
   - Indicates algorithm outperforming or potential overfitting
   - May suggest the measurement is particularly well-suited for the model

3. **Exceptionally Low Scores**: Composite score ≤ 0.3
   - Flagged as `poor_fit`
   - Indicates poor match quality
   - May suggest measurement quality issues or parameter range limitations

4. **No Good Fit Found**: Best composite score ≤ 0.4
   - Flagged as `no_good_fit_found`
   - Indicates the measurement may not be valid for analysis
   - Suggests need to expand parameter ranges or check measurement quality

### Client Accepted Ranges (RGT's Implementation)

RGT defines "client accepted ranges" that represent the client's business-defined acceptable parameter limits. These are used for flagging results that fall outside acceptable limits, but the search space is expanded beyond these ranges to allow discovery of optimal values:

**Client Accepted Ranges**:
- **Lipid**: 9-250nm (client's accepted range)
- **Aqueous**: 800-12000nm (client's accepted range)
- **Roughness**: 600-2750Å (client's accepted range)

**Search Parameter Ranges** (expanded for exploration):
- **Lipid**: 0-500nm (expanded beyond client's 9-250nm to allow discovery)
- **Aqueous**: -20-12000nm (expanded beyond client's 800-12000nm, includes negative values)
- **Roughness**: 0-3000Å (expanded beyond client's 600-2750Å to allow discovery)

Results outside the client accepted ranges are flagged but not excluded from the search, allowing the system to discover optimal values that may lie outside the client's initial expectations.

```yaml
analysis:
  edge_case_detection:
    client_accepted_ranges:
      lipid: {min: 9, max: 250}      # Client's accepted range
      aqueous: {min: 800, max: 12000} # Client's accepted range
      roughness: {min: 600, max: 2750} # Client's accepted range
```

**Rationale**: 
- Yash's initial ranges (H0) may not capture all valid parameter combinations
- During coarse-fine refinement, the search expands beyond initial ranges
- Acceptable ranges define the upper bounds of what's physically/biologically feasible
- If optimal values are found outside acceptable ranges, they're flagged for review

### Flagging Behavior

- **Visual Indicators**: Edge cases in displayed results show a ⚠️ symbol in the results table
- **Warnings**: Only shown if edge cases appear in the top displayed results (e.g., top 10)
- **Details**: Expandable section shows full edge case information for each flagged candidate
- **Non-Blocking**: Flags are informational - they don't prevent the search from running or results from being used

### Example Edge Case Scenarios

1. **Parameter Outside Client Accepted Range**:
   - Result: `lipid=300nm` (outside client's 9-250nm range)
   - Flag: `lipid_outside_client_range(300nm, client_range: 9-250nm)`
   - Action: Review if this is within client's accepted ranges; may indicate need to validate the result

2. **No Good Fit**:
   - Result: Best score = 0.35
   - Flag: `no_good_fit_found`
   - Action: Check measurement quality, expand search ranges, or mark measurement as invalid

3. **Exceptional Performance**:
   - Result: Composite score = 0.95
   - Flag: `exceptional_score`
   - Action: Review if this is legitimate or indicates overfitting

---

## Best Score Selection

### Ranking Process

1. **Evaluation**: All candidates are evaluated and scored using RGT's multi-metric system
2. **Sorting**: Results are sorted by `composite_score` in **descending order** (highest first)
3. **Selection**: The candidate with the **highest composite_score** is ranked #1 (best)

### Result Format

Each result contains:
- **Parameters**: `lipid_nm`, `aqueous_nm`, `roughness_A`
- **Individual Scores**: `score_peak_count`, `score_peak_delta`, `score_phase_overlap`, `score_residual`, etc.
- **Composite Score**: `score_composite` (used for ranking)
- **Diagnostics**: Detailed metrics (RMSE, peak counts, etc.)
- **Edge Case Flags**: `edge_case_flag` (True/False), `edge_case_reason` (description)

### Example Results from Actual Grid Search

The following tables show actual results from grid search runs on sample measurements:

#### Example 1: Good Fit Sample (spectra_09-46-28-833)

| Rank | ⚠️ | Lipid (nm) | Aqueous (nm) | Roughness (Å) | Composite | Peak Count | Peak Delta | Phase | Residual |
|------|---|------------|--------------|---------------|-----------|------------|------------|-------|----------|
| 1    |   | 200        | 670          | 2300          | 0.5592    | 0.50       | 0.60       | 0.67  | 0.75     |
| 2    |   | 205        | 670          | 2225          | 0.5579    | 0.50       | 0.59       | 0.67  | 0.75     |
| 3    |   | 200        | 670          | 2275          | 0.5577    | 0.50       | 0.60       | 0.65  | 0.75     |
| 4    |   | 205        | 670          | 2275          | 0.5572    | 0.50       | 0.58       | 0.69  | 0.75     |
| 5    |   | 205        | 670          | 2250          | 0.5541    | 0.50       | 0.57       | 0.68  | 0.75     |

**Analysis**: This measurement shows good fits with composite scores around 0.55-0.56. The best fit has parameters L=200nm, A=670nm, R=2300Å. Peak count score is 0.50 (4 out of 8 peaks matched), indicating moderate peak matching.

#### Example 2: Good Fit Sample (spectra_09-46-32-072)

| Rank | ⚠️ | Lipid (nm) | Aqueous (nm) | Roughness (Å) | Composite | Peak Count | Peak Delta | Phase | Residual |
|------|---|------------|--------------|---------------|-----------|------------|------------|-------|----------|
| 1    |   | 205        | 690          | 2500          | 0.5269    | 0.38       | 0.55       | 0.84  | 0.72     |
| 2    |   | 200        | 690          | 2500          | 0.5259    | 0.38       | 0.56       | 0.79  | 0.72     |
| 3    |   | 200        | 660          | 2650          | 0.4992    | 0.25       | 0.56       | 0.82  | 0.72     |
| 4    |   | 200        | 670          | 2625          | 0.4988    | 0.25       | 0.56       | 0.82  | 0.72     |
| 5    |   | 200        | 660          | 2625          | 0.4988    | 0.25       | 0.56       | 0.82  | 0.72     |

**Analysis**: This measurement shows moderate fits with composite scores around 0.50-0.53. The best fit has parameters L=205nm, A=690nm, R=2500Å. Phase overlap is high (0.79-0.84), indicating good spectral shape matching, but peak matching is lower (0.25-0.38).

### Handling Ties

If multiple candidates have identical composite scores, they are ranked by:
1. Composite score (primary)
2. Residual score (secondary)
3. Phase overlap score (tertiary)
4. Parameter order (deterministic tie-breaker)

---

## Result Columns Reference

This section explains all columns in the grid search results output. Each row represents one parameter combination that was evaluated.

### Parameter Columns

- **`lipid_nm`**: Lipid layer thickness in nanometers. Typical range: 120-200nm (Yash's initial range), can expand to 0-500nm (acceptable range).
- **`aqueous_nm`**: Aqueous layer thickness in nanometers. Typical range: 680-1200nm (Yash's initial range), can expand to 0-2000nm (acceptable range). Note: Can be 0nm in pathological cases.
- **`roughness_A`**: Mucus layer surface roughness in Angstroms (Å). Typical range: 600-2600Å (Yash's initial range), can expand to 0-5000Å (acceptable range).

### Score Columns (0.0 = worst, 1.0 = best)

- **`score_composite`**: **Primary ranking metric**. Weighted average of all individual metric scores. Higher is better. Used to rank candidates.
- **`score_peak_count`**: Measures how well the number of peaks matches between measured and theoretical spectra. 1.0 = perfect match, 0.0 = no match.
- **`score_peak_delta`**: Measures wavelength alignment accuracy of matched peaks. 1.0 = perfect alignment, 0.0 = poor alignment.
- **`score_phase_overlap`**: Measures similarity in frequency domain (FFT) of detrended spectra. 1.0 = identical frequency content, 0.0 = different characteristics.
- **`score_residual`**: Measures direct fit quality (RMSE-based). 1.0 = perfect fit (RMSE ≈ 0), 0.0 = poor fit (high RMSE).

### Peak Count Diagnostics

- **`peak_count_measurement_peaks`**: Total number of peaks detected in the measured spectrum.
- **`peak_count_theoretical_peaks`**: Total number of peaks detected in the theoretical spectrum.
- **`peak_count_matched_peaks`**: Number of peaks successfully matched between measured and theoretical spectra (within wavelength tolerance).

### Peak Delta Diagnostics

- **`peak_delta_matched_pairs`**: Number of peak pairs that were successfully matched (same as `peak_count_matched_peaks`).
- **`peak_delta_mean_delta_nm`**: Average wavelength difference (in nanometers) between matched peak pairs. Lower is better (0nm = perfect alignment).
- **`peak_delta_unpaired_measurement`**: Number of peaks in the measured spectrum that couldn't be matched to any theoretical peak.
- **`peak_delta_unpaired_theoretical`**: Number of peaks in the theoretical spectrum that couldn't be matched to any measured peak.

### Phase Overlap Diagnostics

- **`phase_overlap_coherence`**: Absolute value of the dot product between FFT coefficients of measured and theoretical spectra. Higher indicates better frequency domain similarity.
- **`phase_overlap_norm_reference`**: Norm (magnitude) of the measured spectrum's FFT coefficients.
- **`phase_overlap_norm_candidate`**: Norm (magnitude) of the theoretical spectrum's FFT coefficients.

### Residual Diagnostics

- **`residual_rmse`**: Root Mean Square Error between measured and theoretical spectra. Lower is better (0 = perfect fit). Typical good values: < 0.01.
- **`residual_mae`**: Mean Absolute Error between measured and theoretical spectra. Lower is better. Typically smaller than RMSE.
- **`residual_r2`**: R-squared (coefficient of determination). Measures how well the theoretical spectrum explains the variance in the measured spectrum. Higher is better (1.0 = perfect explanation, 0 = no explanation, negative = worse than baseline).
- **`residual_mape_pct`**: Mean Absolute Percentage Error (%). Lower is better. Values > 100% indicate large relative errors.

### Edge Case Flag Column

- **`⚠️`**: Visual indicator (⚠️ symbol) if the result is flagged as an edge case. Empty if no edge case detected.
- **Edge cases include**:
  - Parameters outside client accepted ranges (e.g., lipid > 250nm when client range is 9-250nm)
  - Exceptionally high scores (≥ 0.9) - may indicate overfitting
  - Exceptionally low scores (≤ 0.3) - poor fit quality
  - "No good fit found" - best score ≤ 0.4 threshold

### Interpreting Results

**Good Fit Indicators**:
- `score_composite` > 0.7: Excellent match
- `score_composite` 0.5-0.7: Good match, review individual metrics
- `residual_rmse` < 0.01: Low fit error
- `residual_r2` > 0: Theoretical spectrum explains variance
- `peak_count_matched_peaks` close to `peak_count_measurement_peaks`: Good peak matching

**Warning Signs**:
- `score_composite` < 0.5: Poor match, may need to expand parameter ranges
- `residual_rmse` > 0.1: High fit error
- `residual_r2` < 0: Theoretical spectrum worse than baseline
- Many unpaired peaks: Significant peak structure mismatch
- ⚠️ flag present: Review edge case reason

---

## Application Screenshots

This section will contain screenshots of the Streamlit web application interface to help clients visualize the system in action.

### Spectrum Analysis Page

*[Screenshot to be added: Shows the main spectrum comparison view with theoretical and measured spectra overlaid, parameter sliders, and measurement file selection]*

**Description**: The Spectrum Analysis page allows users to:
- Adjust tear film parameters (lipid thickness, aqueous thickness, roughness) using sliders
- View theoretical spectra generated in real-time
- Compare theoretical spectra with measured experimental data
- Select different measurement files from the sample data

### Detrended Analysis Page

*[Screenshot to be added: Shows detrended spectra, peak detection results, and FFT analysis]*

**Description**: The Detrended Analysis page provides:
- Detrended versions of both theoretical and measured spectra (baseline removed)
- Peak detection visualization showing detected peaks in both spectra
- FFT (frequency domain) analysis showing phase overlap
- Analysis parameter controls (detrending cutoff frequency, peak prominence threshold)

### Grid Search Page

*[Screenshot to be added: Shows grid search interface with strategy selection, results table, and edge case warnings]*

**Description**: The Grid Search page enables:
- Selection of search strategy (coarse-fine recommended, random, or systematic)
- Configuration of search parameters and limits
- Execution of grid search with progress indicators
- Results table showing top candidates ranked by composite score
- Edge case detection and flagging (⚠️ indicators)
- Ability to apply best-fit parameters to sliders for further analysis

---

## Configuration

### Strategy Selection

In the Streamlit UI, select search strategy via dropdown:
- **"random"**: Random or systematic search (auto-selected based on coverage)
- **"coarse-fine"**: Two-stage coarse-to-fine search (default, recommended)

### Parameter Ranges (Yash's H0)

Configured in `config.yaml` → `parameters`:
```yaml
parameters:
  lipid:
    min: 120        # Yash's initial range (H0)
    max: 200
    step: 5
  aqueous:
    min: 680        # Yash's initial range (H0)
    max: 1200
    step: 10
  roughness:
    min: 600        # Yash's initial range (H0)
    max: 2600
    step: 50
```

### Coarse-Fine Configuration

```yaml
analysis:
  grid_search:
    coarse:
      top_k: 20                    # Number of top candidates to refine
      lipid: {min: 120, max: 200, step: 20}
      aqueous: {min: 680, max: 1200, step: 100}
      roughness: {min: 600, max: 2600, step: 400}
    refine:
      lipid: {window_nm: 20, step_nm: 5}      # ±10nm window, 5nm steps
      aqueous: {window_nm: 40, step_nm: 10}   # ±20nm window, 10nm steps
      roughness: {window_A: 200, step_A: 25}  # ±100Å window, 25Å steps
      max_refine_candidates: 5000             # Budget limit for refinement
```

### Edge Case Detection Configuration

```yaml
analysis:
  edge_case_detection:
    enabled: true
    threshold_high_score: 0.9    # Flag scores >= this as exceptional
    threshold_low_score: 0.3     # Flag scores <= this as poor fits
    threshold_no_fit: 0.4        # If best score <= this, flag as "no good fit"
    acceptable_ranges:
      lipid: {min: 0, max: 500}      # Wider than Yash's search range
      aqueous: {min: 0, max: 2000}    # Wider than Yash's search range
      roughness: {min: 0, max: 5000}  # Wider than Yash's search range
```

### Metric Weights

```yaml
metrics:
  composite:
    weights:
      residual: 0.5
      peak_count: 0.4
      peak_delta: 0.4
      phase_overlap: 0.2
      quality: 0.1
      temporal_continuity: 0.1
```

**Recommendation**: Adjust weights based on which aspects of the fit are most important for your use case.

---

## Results Documentation Template

This section provides a template for documenting grid search results, addressing the requirement for "Results Documentation Template" in the project requirements.

### Summary Statistics

For each grid search run, document:

1. **Spectrum Fitting Success Rate**:
   - Number of spectra successfully fitted vs. total evaluated
   - Success rate: `(successful_fits / total_evaluated) × 100%`
   - Comparison to Yash's baseline:
     - Number of successful fits vs. originally successful (if known)
     - Number of successful fits vs. originally failed (if known)

2. **Performance Metrics**:
   - **Time per spectrum**: Average evaluation time (typically ~0.025 seconds)
   - **Time per complete test**: Total execution time for full grid search
   - **Combinations evaluated**: Total number of parameter combinations tested
   - **Search strategy used**: random, systematic, or coarse-fine

3. **Edge Case Statistics**:
   - **Total edge cases detected**: Number of flagged candidates
   - **Edge case breakdown**: Count by type:
     - Parameters outside client accepted ranges
     - Exceptionally high scores
     - Exceptionally low scores
     - No good fit found
   - **Edge cases in top results**: Number of flagged candidates in displayed top-K

4. **Parameter Range Analysis**:
   - **Explored ranges**: Min/max values actually tested for each parameter
   - **Optimal parameter paths**: Parameter combinations that led to successful fits
   - **Boundary cases**: How often optimal values were at search boundaries (suggests need to expand ranges)

5. **System Behavior**:
   - **Cases where system "went wild"**: Instances where the search explored many combinations without finding good fits
   - **Search efficiency**: Percentage of search space covered vs. time taken
   - **Refinement effectiveness**: For coarse-fine, how many refined candidates improved upon coarse results

### Example Results Documentation

#### Example 1: Good Fit Sample (spectra_09-46-28-833)

```
Grid Search Results - Measurement: good_fit/9/spectra_09-46-28-833

Date: 2024-11-20
Strategy: coarse-fine
Total Combinations Evaluated: ~4,680
Execution Time: ~2 minutes
Time per Spectrum: ~0.025 seconds

Success Metrics:
- Best Composite Score: 0.5592
- Top 5 Scores Range: 0.5446 - 0.5592
- All Top 10 Scores: > 0.5 (good fits)

Edge Cases:
- Total Edge Cases: 0 (no flags in top 10)
- Parameters Outside Client Accepted Ranges: 0
- Exceptionally High Scores (≥0.9): 0
- Exceptionally Low Scores (≤0.3): 0
- No Good Fit Found: No (best score 0.5592 > 0.4 threshold)

Parameter Ranges Explored:
- Lipid: 200-210nm (within initial 120-200nm range)
- Aqueous: 660-680nm (within initial 680-1200nm range)
- Roughness: 2200-2300Å (within initial 600-2600Å range)

Optimal Parameters (Top 5):
1. L=200nm, A=670nm, R=2300Å, Score=0.5592
   - Peak Count: 0.50 (4/8 peaks matched)
   - Peak Delta: 0.60 (mean delta: 1.53nm)
   - Phase Overlap: 0.67
   - Residual RMSE: 0.0043, R²: 0.0034
2. L=205nm, A=670nm, R=2225Å, Score=0.5579
3. L=200nm, A=670nm, R=2275Å, Score=0.5577
4. L=205nm, A=670nm, R=2275Å, Score=0.5572
5. L=205nm, A=670nm, R=2250Å, Score=0.5541

Notes:
- Good peak matching: 4 out of 8 measurement peaks matched
- Consistent parameter region: All top results cluster around L=200-205nm, A=670nm, R=2200-2300Å
- Moderate composite scores (0.55-0.56) indicate acceptable but not exceptional fits
- Residual RMSE values are low (~0.004), indicating good point-by-point fit quality
```

#### Example 2: Good Fit Sample (spectra_09-46-32-072)

```
Grid Search Results - Measurement: good_fit/10/spectra_09-46-32-072

Date: 2024-11-20
Strategy: coarse-fine
Total Combinations Evaluated: ~4,680
Execution Time: ~2 minutes
Time per Spectrum: ~0.025 seconds

Success Metrics:
- Best Composite Score: 0.5269
- Top 5 Scores Range: 0.4945 - 0.5269
- All Top 10 Scores: > 0.49 (moderate to good fits)

Edge Cases:
- Total Edge Cases: 0 (no flags in top 10)
- Parameters Outside Client Accepted Ranges: 0
- Exceptionally High Scores (≥0.9): 0
- Exceptionally Low Scores (≤0.3): 0
- No Good Fit Found: No (best score 0.5269 > 0.4 threshold)

Parameter Ranges Explored:
- Lipid: 200-205nm (within initial 120-200nm range)
- Aqueous: 660-690nm (within initial 680-1200nm range)
- Roughness: 2500-2700Å (expanded beyond initial 600-2600Å range during refinement)

Optimal Parameters (Top 5):
1. L=205nm, A=690nm, R=2500Å, Score=0.5269
   - Peak Count: 0.38 (3/8 peaks matched)
   - Peak Delta: 0.55 (mean delta: 1.64nm)
   - Phase Overlap: 0.84 (excellent spectral shape match)
   - Residual RMSE: 0.0049, R²: 0.0062
2. L=200nm, A=690nm, R=2500Å, Score=0.5259
3. L=200nm, A=660nm, R=2650Å, Score=0.4992
4. L=200nm, A=670nm, R=2625Å, Score=0.4988
5. L=200nm, A=660nm, R=2625Å, Score=0.4988

Notes:
- Lower peak matching: Only 2-3 out of 8 measurement peaks matched
- Excellent phase overlap (0.79-0.84) compensates for lower peak matching
- Refinement expanded roughness range to 2650-2700Å (beyond initial 2600Å max)
- Residual RMSE values are low (~0.005), indicating good fit quality despite lower peak scores
- Composite score dominated by phase_overlap and residual metrics
```

---

## Performance Considerations

### Execution Time

**Per-spectrum evaluation time**: ~0.025 seconds (40 spectra/second)

**Full Search (36,941 combinations - Yash's ranges)**:
- Time: ~36,941 × 0.025s = **~15-30 minutes**
- Memory: ~100MB (results DataFrame)

**Coarse-Fine Search**:
- Coarse: 180 combinations × 0.025s = **4.5 seconds**
- Refine: ~4,500 combinations × 0.025s = **112.5 seconds**
- **Total: ~2 minutes** (87% faster)

**Random Search (1,000 samples)**:
- Time: 1,000 × 0.025s = **25 seconds**

### Memory Usage

- **Per candidate**: ~2KB (parameters + scores + diagnostics)
- **10,000 candidates**: ~20MB
- **Full search (36,941)**: ~74MB

### Optimization Tips

1. **Use Coarse-Fine**: 87% time reduction with similar quality
2. **Adjust Step Sizes**: Larger steps = fewer combinations = faster
3. **Set max_results**: Limit evaluation budget for quick tests
4. **Disable Unused Metrics**: Reduces computation per candidate
5. **Use CLI for Batch Processing**: More efficient than UI for multiple measurements

---

## Known Limitations and Future Work

### Current Limitations

1. **Parameter Range Discovery**: 
   - The system may miss optimal values if they're significantly outside Yash's initial ranges (120-200nm, 680-1200nm, 600-2600Å)
   - **Status**: Partially addressed - coarse-fine search expands during refinement, but initial coarse search is limited to configured ranges
   - **Future Work**: Need to identify with Yash or AdOM's team which parameters may be missing

2. **Fit Percentage**: 
   - Current fit percentages may not be as high as desired
   - **Possible Causes**:
     - Parameter ranges may need expansion
     - Additional parameters may need to be varied (currently only 3 parameters)
     - Measurement quality issues
     - Metric weights may need adjustment
   - **Future Work**: Analyze which parameter ranges/paths lead to successful fits vs. failures

3. **Validation Methodology**: 
   - **Status**: Implemented - edge case detection and flagging system is in place
   - **Future Work**: Collect feedback on flag types to refine detection criteria

4. **Comparison to Yash's System**: 
   - Need to track: Number of successful fits vs. originally successful/failed in Yash's system
   - **Future Work**: Implement comparison metrics to Yash's baseline results

### Future Enhancements

1. **Additional Parameters**: 
   - Consider varying additional parameters beyond the current 3 (lipid, aqueous, roughness)
   - May need coordination with Yash or AdOM team to identify which parameters to add

2. **Adaptive Range Expansion**: 
   - Automatically expand search ranges if optimal values consistently hit boundaries
   - Learn from historical results to adjust initial ranges

3. **Feedback Loop for Flags**: 
   - Collect user feedback on flagged edge cases
   - Refine flagging criteria based on which flags are most useful
   - Track which flags lead to actionable insights

4. **Performance Benchmarking**: 
   - Compare RGT's grid search results to Yash's manual selection results
   - Measure improvement in fit quality and success rate
   - Document time savings vs. manual approach

---

## Summary

### Key Takeaways

1. **Yash's Baseline (H0)**: Established initial parameter ranges (120-200nm, 680-1200nm, 600-2600Å) and web application framework. RGT expanded these ranges (0-500nm, -20-12000nm, 0-3000Å) based on client testing showing optimal values outside algorithm constraints.
2. **RGT's Implementation**: Added intelligent grid search with coarse-fine strategy, multi-metric scoring, and edge case detection
3. **Coarse-Fine Search** is the recommended strategy for balancing speed and quality (87% faster than exhaustive search)
4. **Composite Score** combines 6 metrics using weighted averaging for comprehensive assessment
5. **Best Score** = highest composite_score (ranked #1)
6. **Edge Case Detection** flags parameters outside client accepted ranges, exceptional/poor scores, and "no good fit" scenarios
7. **Full Search** takes ~15-30 minutes; Coarse-Fine takes ~7-8 minutes (optimized for ~20 min total with expanded ranges)

### When to Use Each Strategy

- **Coarse-Fine**: Default choice, best balance of speed/quality, can expand beyond initial ranges
- **Random**: Quick exploration with limited budget (<10% coverage)
- **Systematic/Full**: Need complete coverage, time not a concern

### Interpreting Results

- **Composite Score > 0.7**: Good match, likely reliable
- **Composite Score 0.5-0.7**: Moderate match, review individual metrics
- **Composite Score < 0.5**: Poor match, consider:
  - Expanding parameter ranges
  - Checking measurement quality
  - Reviewing metric weights
  - Verifying theoretical spectrum generation
- **Edge Case Flags**: Review flagged candidates carefully - they may indicate:
  - Need to expand search ranges
  - Measurement quality issues
  - Exceptional performance (verify legitimacy)

---

## Appendix: Mathematical Details

### Peak Matching Algorithm

1. For each measured peak, find closest theoretical peak within tolerance
2. Use greedy matching (first-come-first-served)
3. Each theoretical peak can only match one measured peak
4. Unmatched peaks contribute to penalty

### FFT Phase Overlap

The phase overlap score uses the normalized inner product (cosine similarity) in frequency space:

```
score = |⟨X_theo, X_meas⟩| / (||X_theo|| × ||X_meas||)
```

Where `X_theo` and `X_meas` are the FFT coefficients of detrended spectra.

### Residual Score Decay

The residual score uses exponential decay:

```
score = exp(-RMSE / tau_rmse)
```

This gives:
- **High scores** for low RMSE (good fit)
- **Rapid decay** as RMSE increases
- **Tunable sensitivity** via `tau_rmse` parameter

---

*Documentation Version: 2.0*  
*Last Updated: 2024*  
*Developed by RGT, building upon Yash's initial web application framework*

