# Peak Detection Findings (measured vs theoretical spectra)

## Sources
- Detection constants for amplitude RMS and nominal thickness mapping  
```1:23:TFI_Common_FINAL/Binaries/SpectrumAmplitudeAnalysis_Config.xml
<app_config>
  <PeaksDetectionType val="Normal" />
  <RangeNormal min="760" max="1070" />
  <RangeSmall min="750" max="1070" />
  <MovingAverageWidthForPeaksDetectionNm min="9" max="27" />
  <NumberOfTimesForMovingAveragings val="2" />
  <AmplitudeRmsThreshold val="0.0012" />
  <FormulaCoefficientsForMinDistanceBetweenAdjacentPeaks constant1="11" denominator1="140" constant2="7" denominator2="70" />
  <MinIntensityBetweenAdjacentPeaks val1="0.00001" val2="0.00002" />
  <FormulaCoefficientsForNominalThicknessNormal constant="-628" slope="485" />
  <FormulaCoefficientsForNominalThicknessSmall constant="-866" slope="780" />
</app_config>
```
- Runtime analysis defaults (applied both to measured and theoretical unless overridden)  
```174:218:TFI_Common_FINAL/Data/analysis_config.xml
<SpAnalyzer>
  <WaveLenghtsRangeForSpectrumAmplitudeRMS>750,920</WaveLenghtsRangeForSpectrumAmplitudeRMS>
  <MovingAverageWidthNm>22.6778,67.144</MovingAverageWidthNm>
  <WaveLenghtsRangeForPeaksDetectionNormal>760,1070</WaveLenghtsRangeForPeaksDetectionNormal>
  <WaveLenghtsRangeForPeaksDetectionSmall>750,940</WaveLenghtsRangeForPeaksDetectionSmall>
  <OffsetForWaveLenghtsRangeForPeaksDetection>10</OffsetForWaveLenghtsRangeForPeaksDetection>
  <MovingAverageWidthForPeaksDetectionNm>9,27</MovingAverageWidthForPeaksDetectionNm>
  <NumberOfTimesForMovingAveragings>2</NumberOfTimesForMovingAveragings>
  <FormulaCoefficientsForMinDistanceBetweenAdjacentPeaks>11,140,7,70</FormulaCoefficientsForMinDistanceBetweenAdjacentPeaks>
  <MinIntensityBetweenAdjacentPeaks>0.00001,0.00002</MinIntensityBetweenAdjacentPeaks>
  <FormulaCoefficientsForNominalThicknessNormal>-628,485</FormulaCoefficientsForNominalThicknessNormal>
  <FormulaCoefficientsForNominalThicknessSmall>-866,780</FormulaCoefficientsForNominalThicknessSmall>
  <PercentToReduceInvWaveLengthsDifsRmsForPeaksUniformity>20</PercentToReduceInvWaveLengthsDifsRmsForPeaksUniformity>
  <MultiplierForTSLibStepForSearchNearBestNode>3</MultiplierForTSLibStepForSearchNearBestNode>
</SpAnalyzer>
```
- Test runs used for validation (Regression and TSLibrary)  
```51:86:TFI_Common_FINAL/Tests_Reading_Center_Side/Interpretation_3_TSLibrary_PeakDetectionNormal_WithAndWithoutGOP/LayerThicknessAnalysis_Config.xml
<PeaksDetectionType val="Normal" />
<IsCalculateGoodnessOfPeaks val="1" />
<GoodnessOfPeaksThreshold val="75" />
<WaveLenghtsRangeForPeaksDetectionNormal min="760" max="1070" />
<WaveLenghtsRangeForPeaksDetectionSmall min="750" max="940" />
<OffsetForWaveLenghtsRangeForPeaksDetection val="10" />
<MovingAverageWidthForPeaksDetectionNm min="9" max="27" />
<NumberOfTimesForMovingAveragings val="2" />
<FormulaCoefficientsForMinDistanceBetweenAdjacentPeaks constant1="10" denominator1="200" constant2="5" denominator2="70" />
<MinIntensityBetweenAdjacentPeaks val1="0.00001" val2="0.00002" />
```

## Shared peak-picking parameters
- Detection type: `Normal` or `Small` (or `None` to skip). Range chosen accordingly; optional ±offset expands the window.
- Smoothing before peak search: moving average width 9–27 nm applied twice; additional broader smoothing for spectrum RMS (22.6778–67.144 nm).
- Acceptance thresholds: amplitude RMS > 0.0012; GOP (goodness-of-peaks) must be ≤ threshold (typically 75, occasionally relaxed to 1000 in history runs).
- Peak spacing: enforce `(λ_cur - λ_prev) > constant + λ_prev / denominator` (two coefficient sets for same-type vs interleaved peaks). Default constants 11/140 and 7/70; some tests use 10/200 and 5/70.
- Peak intensity separation: `|I_cur - I_prev| > 1e-5` (second value unused).
- Nominal thickness estimate from peak count: `thickness = -628 + 485 * n_peaks` for Normal; `-866 + 780 * n_peaks` for Small.
- Uniformity term for GOP: compute inverse wavelength differences between adjacent peaks, drop the top `PercentToReduceInvWaveLengthsDifsRmsForPeaksUniformity`% (20–50), then take RMS; lower GOP is better.

## Measured spectra peak detection
1. Preprocess (if enabled): optional boxcar smoothing (DefaultPreprocessingScheme applies boxcar width 11 twice); optional filtering on AbsoluteDifMaxAbs/Intensity.
2. Compute spectrum amplitude RMS over `WaveLenghtsRangeForSpectrumAmplitudeRMS` (e.g., 720–920 or 750–920 nm); reject if RMS absolute or ratio fails separate quality gates.
3. Smooth amplitude RMS with broad moving average (22–67 nm), then apply peak-detection moving average (9–27 nm) twice.
4. Restrict to detection window: choose Normal or Small range, then expand by offset (e.g., ±10 nm).
5. Find local maxima above amplitude RMS threshold; keep peaks that satisfy min-distance formula and intensity gap. Count peaks.
6. Compute GOP when enabled: uses retained peaks plus uniformity trimming; accept only if GOP ≤ threshold.
7. Derive nominal aqueous thickness from peak count via the linear coefficients.

## Theoretical spectra (TSLibrary) peak detection
- Generate theoretical spectra across thickness grid (Algorithm XML defines parameter ranges); search is widened near best node using `MultiplierForTSLibStepForSearchNearBestNode`.
- Apply the same smoothing and peak-picking parameters as measured spectra (detection type, range, offset, moving-average width, spacing/intensity filters).
- GOP is also computed for theoretical spectra when enabled; comparisons and merit selection use the same threshold logic as measured runs.
- Normalized theoretical spectra are additionally smoothed with boxcar and Gaussian kernels (e.g., width 20 nm, kernel size 7) before comparison.

## Comparator mode
- `PeaksDetectionComparator_Config.xml` supports a mode where ranges come from the algorithm XML (WorkingMode `Comparator`) instead of static detection type; otherwise it uses the config’s explicit ranges (e.g., Normal 760–800 nm with ±10 nm offset). All other spacing/intensity rules remain unchanged.

## Python replication outline
1. Load spectrum (`wavelength_nm`, `intensity`) and optionally apply preprocessing/filtering (boxcar 11 twice; optional AbsoluteDifMaxAbs/Intensity checks).
2. Trim to amplitude RMS range; compute RMS and basic quality gates (absolute and ratio).
3. Smooth intensity with wide moving average (≈23–67 nm) then with detection kernel (≈9–27 nm) repeated twice.
4. Choose detection window (Normal or Small) and apply offset; mask data outside.
5. Identify local maxima; drop any below amplitude threshold.
6. Enforce spacing formula sequentially (use normal or interleaved constants) and intensity gap.
7. Compute GOP: inverse wavelength deltas between accepted peaks → trim top 20–50% → RMS; reject if above threshold.
8. Derive nominal thickness from peak count via the linear formula; use the same logic for theoretical spectra generated from the library.

These steps mirror the parameters observed in the configs above and match the logged runs in `Tests_Reading_Center_Side/Output` where spectra list a peak count and GOP for acceptance.

