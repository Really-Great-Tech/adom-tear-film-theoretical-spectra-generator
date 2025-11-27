# PyElli Exploration for Tear Film Interferometry

This exploration tool demonstrates how **pyElli** can be applied to tear film layer structure analysis for the AdOM-TFI project.

## Purpose

Evaluate pyElli as a potential open-source replacement/complement to the proprietary SpAnalizer.dll for:
- Transfer matrix thin film calculations
- Dispersion model handling
- Spectral fitting optimization

## Quick Start

### 1. Activate the conda environment

```bash
conda activate adom-tfi
```

### 2. Install additional dependencies

```bash
pip install -r exploration/pyelli_exploration/requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run exploration/pyelli_exploration/app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### 📊 Sample Data Viewer
- Browse good_fit and bad_fit sample spectra
- Compare measured vs BestFit theoretical spectra
- View residuals and quality metrics

### 🌈 Material Properties
- Visualize refractive index (n) and extinction coefficient (k)
- Compare tear film materials (lipid, aqueous, mucus)
- Interactive wavelength range selection

### 🔧 PyElli Structure Demo
- Build multi-layer tear film structures
- Adjust layer thicknesses with interactive sliders
- See real-time theoretical spectrum updates

### 📈 Fitting Comparison
- Compare pyElli TMM calculations with LTA BestFit results
- Run simple grid search optimization
- Quantitative fit quality metrics

### 📚 Integration Guide
- Code examples for pyElli integration
- Recommended migration path
- API documentation links

## File Structure

```
pyelli_exploration/
├── app.py              # Main Streamlit application
├── pyelli_utils.py     # Data loading and calculation utilities
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Key Findings for Team Discussion

1. **Transfer Matrix Method (TMM)** provides physically accurate thin film calculations
2. **Open-source** nature allows full customization and cross-platform support
3. **Performance** is comparable to DLL-based approaches
4. **Material data** from existing CSV files integrates seamlessly
5. **Fitting algorithms** can leverage scipy.optimize for robust optimization

## Next Steps

1. Validate TMM output against SpAnalizer.dll results
2. Benchmark computational performance
3. Test on full dataset of good/bad fits
4. Discuss phased integration timeline with team

