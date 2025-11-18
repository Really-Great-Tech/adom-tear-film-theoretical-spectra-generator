# Spectra Quality Components Analysis
## Based on Interpretation History

This document summarizes all components that influence good or bad spectra quality in the TFI (Tear Film Interferometry) system, based on analysis of interpretation history and configuration files.

---

## Executive Summary

The quality of spectra in TFI analysis is determined by multiple components working together. Good spectra typically exhibit:
- High RMS values relative to maximum
- Appropriate Dif Max Abs values
- Good peak detection (GOP ≤ threshold)
- Proper amplitude characteristics
- Successful preprocessing and filtering

Bad spectra are characterized by:
- Low RMS ratios or absolute values below thresholds
- Poor peak detection or GOP above threshold
- Inadequate amplitude characteristics
- Failed preprocessing/filtering validation

---

## 1. RMS (Root Mean Square) Components

### 1.1 RMS Minimal Ratio from Maximum
- **Parameter**: `rms_minimal_ratio_from_max_in_percents`
- **Purpose**: Minimum RMS value as a percentage of the maximum RMS value
- **Typical Range**: 44-50%
- **Good Spectra**: RMS ratio ≥ threshold (typically 44-50%)
- **Bad Spectra**: RMS ratio < threshold
- **Impact**: Critical - determines if spectrum amplitude is sufficient relative to maximum observed

**Values from History:**
- Default: 50%
- Some configurations: 44.040201%

### 1.2 RMS Minimal Absolute Value
- **Parameter**: `rms_minimal_absolute_value`
- **Purpose**: Absolute minimum RMS threshold regardless of maximum
- **Typical Value**: 0.0007
- **Good Spectra**: RMS absolute value ≥ 0.0007
- **Bad Spectra**: RMS absolute value < 0.0007
- **Impact**: Critical - absolute floor for spectrum quality

**Values from History:**
- Default: 0.0007
- Consistent across configurations

### 1.3 Wave Lengths Range for Spectrum Amplitude RMS
- **Parameter**: `wave_lenghts_range_for_spectrum_amplitude_rms`
- **Purpose**: Wavelength range used for RMS calculation
- **Typical Range**: 720-920 nm or 750-920 nm
- **Good Spectra**: RMS calculated within proper wavelength range
- **Bad Spectra**: RMS calculated outside optimal range
- **Impact**: High - determines which spectral region is analyzed

**Values from History:**
- Default: 750-920 nm
- Some configurations: 720-920 nm
- Runtime: 780-1000 nm

---

## 2. Dif Max Abs (Difference Maximum Absolute) Components

### 2.1 Dif Max Abs Minimal Ratio from Maximum
- **Parameter**: `dif_max_abs_minimal_ratio_from_max_in_percents`
- **Purpose**: Minimum difference maximum absolute value as percentage of maximum
- **Typical Range**: 50%
- **Good Spectra**: Dif Max Abs ratio ≥ 50%
- **Bad Spectra**: Dif Max Abs ratio < 50%
- **Impact**: High - measures spectral variation quality

**Values from History:**
- Default: 50%
- Consistent across configurations

### 2.2 Dif Max Abs Minimal Absolute Value
- **Parameter**: `dif_max_abs_minimal_absolute_value`
- **Purpose**: Absolute minimum threshold for difference maximum absolute value
- **Typical Values**: 0.0001 - 0.000111
- **Good Spectra**: Dif Max Abs absolute value ≥ threshold
- **Bad Spectra**: Dif Max Abs absolute value < threshold
- **Impact**: High - absolute floor for spectral variation

**Values from History:**
- Default: 0.0001
- Some configurations: 0.000103
- Some configurations: 0.000111

---

## 3. Goodness of Peaks (GOP) Components

Note: In this pipeline, GOP is a lower-is-better metric. The `GOPThreshold` is a maximum acceptance cap; spectra pass the GOP check when GOP ≤ `GOPThreshold`. Increasing the threshold loosens the GOP constraint.

### 3.1 GOP Calculation Flag
- **Parameter**: `IsCalculateGoodnessOfPeaks` / `IsCalculateGOP`
- **Purpose**: Enable/disable GOP calculation
- **Values**: 0 (disabled) or 1 (enabled)
- **Good Spectra**: GOP calculated and at or below threshold
- **Bad Spectra**: GOP above threshold or calculation disabled when needed
- **Impact**: Critical - primary quality metric for peak detection

**Values from History:**
- Default: 1 (enabled)
- Consistent across configurations

### 3.2 GOP Threshold
- **Parameter**: `GoodnessOfPeaksThreshold` / `GOPThreshold`
- **Purpose**: Maximum GOP value for acceptable spectra
- **Typical Range**: 75-1000
- **Good Spectra**: GOP ≤ threshold
- **Bad Spectra**: GOP > threshold
- **Impact**: Critical - determines acceptance/rejection of spectra

**Values from History:**
- Default: 75
- Some configurations: 85
- Some configurations: 50
- Interpretation History: 1000 (very high threshold, looser GOP constraint; relies more on other metrics)

### 3.3 Peaks Detection Type
- **Parameter**: `PeaksDetectionType`
- **Purpose**: Type of peak detection algorithm
- **Values**: 'Normal', 'Small', 'None'
- **Good Spectra**: Appropriate detection type for spectrum characteristics
- **Bad Spectra**: Wrong detection type or 'None' when peaks needed
- **Impact**: Critical - determines how peaks are identified

**Values from History:**
- Default: 'Normal'
- Some configurations: 'Small'
- Some configurations: 'None' (disables peak detection)

---

## 4. Peak Detection Parameters

### 4.1 Wave Lengths Range for Peaks Detection (Normal)
- **Parameter**: `WaveLenghtsRangeForPeaksDetectionNormal`
- **Purpose**: Wavelength range for normal peak detection
- **Typical Range**: 760-1070 nm
- **Good Spectra**: Peaks detected within proper range
- **Bad Spectra**: Peaks outside detection range
- **Impact**: High - defines detection window

**Values from History:**
- Default: 760-1070 nm
- Consistent across configurations

### 4.2 Wave Lengths Range for Peaks Detection (Small)
- **Parameter**: `WaveLenghtsRangeForPeaksDetectionSmall`
- **Purpose**: Wavelength range for small peak detection
- **Typical Range**: 750-940 nm
- **Good Spectra**: Small peaks detected within proper range
- **Bad Spectra**: Small peaks outside detection range
- **Impact**: Medium - used for alternative detection mode

**Values from History:**
- Default: 750-940 nm
- Consistent across configurations

### 4.3 Moving Average Width for Peaks Detection
- **Parameter**: `MovingAverageWidthForPeaksDetectionNm`
- **Purpose**: Width of moving average filter for peak detection
- **Typical Range**: 9-27 nm
- **Good Spectra**: Appropriate smoothing for peak detection
- **Bad Spectra**: Over/under smoothing affecting peak detection
- **Impact**: Medium - affects peak detection sensitivity

**Values from History:**
- Default: 9-27 nm
- Some configurations: 13.7845-40.4643 nm

### 4.4 Number of Moving Averagings
- **Parameter**: `NumberOfTimesForMovingAveragings`
- **Purpose**: Number of times moving average is applied
- **Typical Value**: 2
- **Good Spectra**: Appropriate smoothing iterations
- **Bad Spectra**: Too many/few iterations
- **Impact**: Medium - affects noise reduction

**Values from History:**
- Default: 2
- Consistent across configurations

### 4.5 Formula Coefficients for Min Distance Between Adjacent Peaks
- **Parameter**: `FormulaCoefficientsForMinDistanceBetweenAdjacentPeaks`
- **Purpose**: Coefficients for minimum distance formula between peaks
- **Formula**: (LambdaCur - LambdaPrev) > (Const + LambdaPrev / Den)
- **Typical Values**: constant1=11, denominator1=140, constant2=7, denominator2=70
- **Good Spectra**: Peaks properly spaced according to formula
- **Bad Spectra**: Peaks too close together (false positives)
- **Impact**: High - prevents false peak detection

**Values from History:**
- Default: constant1=11, denominator1=140, constant2=7, denominator2=70
- Some configurations: constant1=1, denominator1=140, constant2=1, denominator2=70 (more lenient)
- Some configurations: constant1=10, denominator1=200, constant2=5, denominator2=100

### 4.6 Minimal Intensity Between Adjacent Peaks
- **Parameter**: `MinIntensityBetweenAdjacentPeaks`
- **Purpose**: Minimum intensity difference required between adjacent peaks
- **Formula**: |IntensityCur - IntensityPrev| > Val
- **Typical Values**: 0.00001 (same type), 0.00002 (interleaved)
- **Good Spectra**: Peaks with sufficient intensity separation
- **Bad Spectra**: Peaks with insufficient intensity difference
- **Impact**: Medium - filters weak peaks

**Values from History:**
- Default: 0.00001, 0.00002
- Consistent across configurations

---

## 5. Spectrum Amplitude Components

### 5.1 Spectrum Amplitude Low Threshold
- **Parameter**: `AmplitudeLowTresholdForSpectrum`
- **Purpose**: Minimum amplitude for good spectrum points
- **Typical Value**: 0.00025
- **Good Spectra**: Amplitude ≥ 0.00025
- **Bad Spectra**: Amplitude < 0.00025
- **Impact**: Critical - absolute floor for spectrum amplitude

**Values from History:**
- Default: 0.00025

### 5.2 Spectrum Average Range
- **Parameter**: `wave_lenghts_range_for_spectrum_average`
- **Purpose**: Wavelength range for spectrum average calculation
- **Typical Range**: 820-980 nm
- **Good Spectra**: Average calculated within proper range
- **Bad Spectra**: Average calculated outside optimal range
- **Impact**: Medium - affects average value calculation

**Values from History:**
- Default: 820-980 nm
- Consistent across configurations

### 5.3 Percent Range for Spectrum Average
- **Parameter**: `percent_range_for_spectrum_average`
- **Purpose**: Percent range for spectrum average validation
- **Typical Range**: 1.6-5.6% or 1.759809-2.4431%
- **Good Spectra**: Average within percent range
- **Bad Spectra**: Average outside percent range
- **Impact**: Medium - validates spectrum consistency

**Values from History:**
- Default: 1.6-5.6%
- Some configurations: 1.759809-2.4431%

---

## 6. Preprocessing and Filtering Components

### 6.1 Spectra Preprocessing Scheme
- **Parameter**: `IsUseSpectraPreprocessingScheme`
- **Purpose**: Enable/disable preprocessing scheme
- **Values**: 0 (disabled) or 1 (enabled)
- **Good Spectra**: Proper preprocessing applied
- **Bad Spectra**: Preprocessing disabled when needed, or incorrect preprocessing
- **Impact**: High - affects raw spectrum quality

**Values from History:**
- Default: 1 (enabled)
- Consistent across configurations

### 6.2 Spectra Default Preprocessing Scheme
- **Parameter**: `IsUseSpectraDefaultPreprocessingScheme`
- **Purpose**: Use default vs custom preprocessing
- **Values**: 0 (custom) or 1 (default)
- **Good Spectra**: Appropriate preprocessing scheme
- **Bad Spectra**: Wrong preprocessing scheme
- **Impact**: Medium - affects preprocessing quality

**Values from History:**
- Default: 1 (use default)
- Consistent across configurations

### 6.3 Spectra Filtering Scheme
- **Parameter**: `IsUseSpectraFilteringScheme`
- **Purpose**: Enable/disable filtering scheme
- **Values**: 0 (disabled) or 1 (enabled)
- **Good Spectra**: Proper filtering applied
- **Bad Spectra**: Filtering disabled when needed, or incorrect filtering
- **Impact**: High - affects noise reduction

**Values from History:**
- Default: 1 (enabled)
- Interpretation History: 0 (disabled) - may indicate filtering issues

### 6.4 Spectra Default Filtering Scheme
- **Parameter**: `IsUseSpectraDefaultFilteringScheme`
- **Purpose**: Use default vs custom filtering
- **Values**: 0 (custom) or 1 (default)
- **Good Spectra**: Appropriate filtering scheme
- **Bad Spectra**: Wrong filtering scheme
- **Impact**: Medium - affects filtering quality

**Values from History:**
- Default: 1 (use default)
- Interpretation History: 0 (disabled) - may indicate filtering issues

---

## 7. Smoothing Components

### 7.1 Boxcar Width for Spectra Smoothing
- **Parameter**: `BoxcarWidthForSpectraSmoothing`
- **Purpose**: Width of boxcar filter for smoothing
- **Typical Range**: 11-20 nm
- **Good Spectra**: Appropriate smoothing level
- **Bad Spectra**: Over/under smoothing
- **Impact**: Medium - affects noise reduction vs detail preservation

**Values from History:**
- Default: 11 nm
- Some configurations: 17 nm
- Some configurations: 20 nm

### 7.2 Gaussian Kernel Size for Spectra Smoothing
- **Parameter**: `GaussKernelSizeForSpectraSmoothing`
- **Purpose**: Size of Gaussian kernel for smoothing
- **Values**: 7, 9, or 11
- **Good Spectra**: Appropriate kernel size
- **Bad Spectra**: Wrong kernel size causing artifacts
- **Impact**: Medium - affects smoothing quality

**Values from History:**
- Default: 11
- Some configurations: 7
- Some configurations: 9
- Interpretation History: 11

---

## 8. FFT (Fast Fourier Transform) Components

### 8.1 Wave Lengths Range for FFT
- **Parameter**: `wave_lenghts_range_for_fft`
- **Purpose**: Wavelength range for FFT analysis
- **Typical Range**: 750-1050 nm
- **Good Spectra**: FFT calculated within proper range
- **Bad Spectra**: FFT calculated outside optimal range
- **Impact**: Medium - affects frequency domain analysis

**Values from History:**
- Default: 750-1050 nm
- Consistent across configurations

### 8.2 Half Value from First FFT Peak in Log Scale
- **Parameter**: `half_value_from_first_fft_peak_in_log_scale`
- **Purpose**: Threshold for FFT peak detection
- **Typical Values**: 0.25, 0.3
- **Good Spectra**: FFT peaks above threshold
- **Bad Spectra**: FFT peaks below threshold
- **Impact**: Medium - affects FFT-based analysis

**Values from History:**
- Default: 0.25, 0.3
- Consistent across configurations

### 8.3 Percent to Reduce FFT Largest First Peak Width
- **Parameter**: `percent_to_reduce_fft_largest_first_peak_width`
- **Purpose**: Percentage reduction for FFT peak width
- **Typical Values**: 80-90%
- **Good Spectra**: Appropriate peak width reduction
- **Bad Spectra**: Incorrect peak width handling
- **Impact**: Low - affects FFT peak analysis

**Values from History:**
- Default: 80-90%
- Consistent across configurations

---

## 9. Moving Average Components

### 9.1 Moving Average Width
- **Parameter**: `moving_average_width_nm`
- **Purpose**: Width of moving average filter
- **Typical Range**: 22.6778-67.144 nm
- **Good Spectra**: Appropriate averaging width
- **Bad Spectra**: Over/under averaging
- **Impact**: High - affects overall spectrum smoothing

**Values from History:**
- Default: 22.6778-67.144 nm
- Consistent across configurations

---

## 10. Interpretation History Specific Values

### 10.1 Interpretation Run 2025-11-02_23-53-09
- **GOP Threshold**: 1000 (very high - loose GOP constraint)
- **Peaks Detection Type**: None (disabled)
- **Filtering Scheme**: Disabled
- **Boxcar Width**: 17 nm
- **Gaussian Kernel**: 11
- **RMS Ratio**: 50%
- **RMS Absolute**: 0.0007
- **Dif Max Abs Ratio**: 50%
- **Dif Max Abs Absolute**: 0.000111

**Analysis**: This configuration uses a very high GOP threshold (1000), which is a loose constraint under GOP-as-maximum semantics; with peak detection disabled, it suggests reliance on other quality metrics (RMS, amplitude, etc.).

### 10.2 Interpretation Run 2025-11-03_14-07-14
- **GOP Threshold**: 1000 (very high - loose GOP constraint)
- **Peaks Detection Type**: None (disabled)
- **Filtering Scheme**: Disabled
- **Boxcar Width**: 17 nm
- **Gaussian Kernel**: 11
- **RMS Ratio**: 50%
- **RMS Absolute**: 0.0007
- **Dif Max Abs Ratio**: 50%
- **Dif Max Abs Absolute**: 0.000111

**Analysis**: Same configuration as previous run, indicating a consistent approach with a loose GOP constraint and emphasis on other metrics.

---

## 11. Component Interaction Summary

### Components That Promote Good Spectra:
1. **High RMS values** (both ratio and absolute) - indicates strong signal
2. **Low GOP values** - indicates good peak detection quality
3. **Proper peak detection** - correct type and parameters
4. **Appropriate preprocessing** - cleans raw data effectively
5. **Effective filtering** - reduces noise without losing signal
6. **Optimal smoothing** - balances noise reduction and detail preservation
7. **Correct wavelength ranges** - ensures analysis in optimal spectral region

### Components That Indicate Bad Spectra:
1. **Low RMS values** - insufficient signal strength
2. **High GOP values** - poor peak detection quality
3. **Incorrect peak detection** - wrong type or parameters
4. **Failed preprocessing** - raw data issues
5. **Ineffective filtering** - noise not properly reduced
6. **Over/under smoothing** - artifacts or lost detail
7. **Wrong wavelength ranges** - analysis outside optimal region

---

## 12. Recommendations

### For Good Spectra:
- Ensure RMS ratio ≥ 50% and absolute ≥ 0.0007
- Maintain GOP ≤ threshold (e.g., 75); lower thresholds are stricter
- Use appropriate peak detection type ('Normal' or 'Small')
- Enable preprocessing and filtering schemes
- Use optimal smoothing parameters (Boxcar 11-17 nm, Gaussian kernel 7-11)
- Verify wavelength ranges match measurement conditions

### For Troubleshooting Bad Spectra:
1. Check RMS values first - most critical component
2. Verify GOP calculation and threshold settings
3. Review peak detection parameters and type
4. Validate preprocessing/filtering schemes are enabled and correct
5. Adjust smoothing parameters if artifacts present
6. Confirm wavelength ranges are appropriate for measurement

---

## 13. References

- Configuration files: `LayerThicknessAnalysis_Config.xml`
- Analysis config: `analysis_config.xml`
- Interpretation History: `Tests_Reading_Center_Side/Interpretation_History/`
- GOP files: `GOP.txt` in interpretation results
- Results files: `ResultsForReport.txt` in interpretation results

---

*Document generated based on analysis of interpretation history and configuration files*
*Last updated: Based on interpretation runs from November 2025*

