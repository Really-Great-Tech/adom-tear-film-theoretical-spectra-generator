# Tear Film Theoretical Spectra Generator

This project generates theoretical tear-film reflectance spectra using AdOM's SpAnalizer.dll, with a Python interface for parameter sweeps, data export, and interactive visualization. The system now features YAML configuration for easy parameter management and automatic generation of interactive Plotly plots.

## Project Structure

```
AdOM-TFI/
├── configs/                          # Configuration files
│   ├── Algorithm_eye_test_4.xml      # Algorithm parameters
│   ├── Configuration1.xml            # General configuration
│   └── Stack_eye_test.xml            # Layer stack definition
├── data/                             # Data files
│   ├── Materials/                    # Material optical properties (.csv files)
│   └── wavelengths/                  # Wavelength grid files
│       ├── Wavelengths813.csv
│       ├── Wavelengths1024.csv
│       └── Wavelengths2061.csv
├── src/                              # Source code and libraries
│   ├── tear_film_generator.py        # Main Python script
│   ├── SpAnalizer.dll               # C++ computation library
│   └── BatchUtility.dll             # Additional utilities
├── outputs/                          # Generated results (created automatically)
│   └── spectra_YYYYMMDD_HHMMSS/     # Timestamped output folders
│       ├── grid.npy                 # 4D reflectance data
│       ├── meta.json                # Parameter metadata
│       ├── config_used.yaml         # Configuration snapshot
│       ├── spectra_plot.html        # Interactive spectral plots
│       └── parameter_sweep.html     # Parameter dependency heatmaps
├── config.yaml                      # Main YAML configuration file
├── single_spectrum_config.yaml      # Single spectrum configuration
├── run_tear_film_generator.py       # Multi-parameter grid generator
├── run_tear_film_generator.bat      # Windows batch for grid generation
├── run_single_spectrum.py           # Single spectrum generator
└── run_single_spectrum.bat          # Windows batch for single spectra
```

## Requirements

- Python 3.8+ with packages:
  - numpy
  - pandas  
  - pyyaml (for configuration files)
  - plotly (for interactive plots)
  - pythonnet (for .NET interop - Windows only)
- Windows environment (required for .NET DLL)

## Quick Start

### Generate Parameter Grid (Multiple Spectra)
```bash
python run_tear_film_generator.py
# Generates multiple spectra across parameter ranges
```

### Generate Single Spectrum (Specific Parameters)
```bash
python run_single_spectrum.py
# Generates one spectrum for specific lipid/aqueous values
```

### Windows Batch Files
```cmd
run_tear_film_generator.bat      # Parameter grid generation
run_single_spectrum.bat          # Single spectrum generation
```

### Direct Script Execution
```bash
# Grid generation
python src/tear_film_generator.py --lipid 150 160 2 --aqueous 850 900 10

# Single spectrum
python src/single_spectrum_generator.py --lipid 150 --aqueous 900
```

## Configuration

### YAML Configuration (`config.yaml`)
The main configuration file allows easy parameter adjustment:

```yaml
# Parameter ranges
parameters:
  lipid: {min: 140, max: 180, step: 5}        # nm
  aqueous: {min: 800, max: 1000, step: 25}    # nm  
  roughness: {min: 1800, max: 2200, step: 50} # Å

# Wavelength selection
wavelengths:
  file: "Wavelengths813.csv"  # or Wavelengths1024.csv, Wavelengths2061.csv

# Output options
output:
  format: "npy"           # or "xlsx"
  include_plots: true     # Generate interactive plots
  plot_format: "html"     # or "png", "pdf"

# Presets for common scenarios
active_preset: "quick_test"  # Use predefined parameter set
```

### Single Spectrum Configuration (`single_spectrum_config.yaml`)
For generating individual spectra with specific parameters:

```yaml
# Target parameters
single_spectrum:
  lipid_thickness: 150      # nm
  aqueous_thickness: 900    # nm
  mucus_roughness: 2000     # Å
  
  # Optional comparison spectra
  comparison_spectra:
    enabled: true
    variations:
      - {lipid: 120, aqueous: 900, label: "Thin lipid"}
      - {lipid: 180, aqueous: 900, label: "Thick lipid"}

# Presets for common conditions
active_preset: "healthy_eye"  # healthy_eye, dry_eye_thin_lipid, thick_lipid
```

### Grid Generation Presets
- **quick_test**: Small parameter space for testing (2×2×2 = 8 spectra)
- **medium_scan**: Medium resolution scan (~hundreds of spectra)
- **high_resolution**: High density parameter sweep (thousands of spectra)
- **lipid_focus**: Focus on lipid thickness variation

### Layer Stack (`configs/Stack_eye_test.xml`)
Defines the optical layer structure:
- **Ambient**: Air environment
- **Lipid**: Tear film lipid layer (variable thickness)
- **Aqueous**: Aqueous tear layer (variable thickness)  
- **Mucus**: Mucus layer (fixed thickness, variable roughness)
- **Substrate**: Underlying tissue

### Algorithm Parameters (`configs/Algorithm_eye_test_4.xml`)
Optimization algorithm settings with parameter constraints:
- **Lipid thickness**: 120-200 nm (step: 5 nm, default: 150 nm)
- **Aqueous thickness**: 680-1200 nm (step: 10 nm, default: 850 nm)  
- **Mucus roughness**: 600-2600 Å (step: 50 Å, default: 2500 Å)

**Note**: Stack defaults may exceed algorithm constraints (e.g., stack roughness 2700Å > algorithm max 2600Å). The system automatically uses algorithm-compliant values with warnings.

### General Configuration (`configs/Configuration1.xml`)
- Materials directory path
- Wavelength file selection
- Aperture settings
- Smearing parameters

## Parameter Ranges

The generator sweeps over three physical parameters, constrained by Algorithm XML limits:

1. **Lipid Layer Thickness** (nm)
   - **Algorithm range**: 120-200 nm (step: 5 nm)
   - Default: 150 nm
   - Current preset: Around 129 nm (requested baseline)

2. **Aqueous Layer Thickness** (nm)  
   - **Algorithm range**: 680-1200 nm (step: 10 nm)
   - Default: 850 nm
   - Current preset: 1200 nm (max limit, since 3194 nm exceeds constraints)

3. **Mucus Layer Roughness** (Å)
   - **Algorithm range**: 600-2600 Å (step: 50 Å)
   - Default: 2500 Å
   - Current preset: 2600 Å (max limit, since stack default 2700 Å exceeds constraints)

## Command Line Options

```bash
python src/tear_film_generator.py [OPTIONS]

Options:
  --dll PATH                Path to SpAnalizer.dll
  --algorithm PATH          Path to algorithm XML file  
  --config PATH             Path to configuration XML file
  --materials PATH          Path to materials directory
  --wavelengths PATH        Path to wavelengths CSV file
  --base_stack PATH         Path to base stack XML file
  --lipid MIN MAX STEP      Lipid thickness range (nm)
  --aqueous MIN MAX STEP    Aqueous thickness range (nm)  
  --rough MIN MAX STEP      Roughness range (Å)
  --out {npy|xlsx}          Output format
  --out_dir PATH            Output directory
```

## Output Files

Results are saved in timestamped directories under `outputs/`:

- **grid.npy**: 4D NumPy array `[lipid, aqueous, rough, wavelength]`
- **grid.xlsx**: Excel file with multiple sheets (if xlsx format selected)
- **meta.json**: Metadata including parameter ranges and wavelengths
- **config_used.yaml**: Snapshot of configuration used for generation
- **spectra_plot.html**: Interactive plot of selected spectra
- **parameter_sweep.html**: Heatmaps showing parameter dependencies

### Interactive Plots
- **Spectral Plot**: Shows reflectance vs wavelength for sample parameter combinations
- **Parameter Sweep**: Heatmaps of reflectance at key wavelengths vs lipid/aqueous thickness
- All plots are interactive with zoom, pan, and hover functionality

## Material Files

The `data/Materials/` directory contains optical property files for various materials:
- Water at different refractive indices
- Lipid properties  
- Tissue optical properties (struma)
- Standard optical materials (BK7, fused silica, etc.)

Each material file is a CSV with wavelength (nm) and optical properties.

## Examples

### Single Spectrum Generation

#### Quick Single Spectrum (uses defaults)
```bash
python run_single_spectrum.py
# Uses 'healthy_eye' preset: L=100nm, A=800nm with comparison spectra
```

#### Custom Parameters via Command Line
```bash
python src/single_spectrum_generator.py --lipid 150 --aqueous 900 --roughness 2200
```

#### Edit Configuration for Different Conditions
Edit `single_spectrum_config.yaml`:
```yaml
active_preset: "dry_eye_thin_lipid"  # or "thick_lipid", "thin_aqueous"
```

### Grid Generation

#### Quick Test Grid
```bash
python run_tear_film_generator.py
# Uses 'quick_test' preset: 2×2×2 = 8 spectra with plots
```

#### Use Different Grid Preset
Edit `config.yaml` and change:
```yaml
active_preset: "medium_scan"  # or "high_resolution", "lipid_focus"
```

#### Command Line Override for Grid
```bash
python src/tear_film_generator.py \
  --lipid 140 180 5 \
  --aqueous 800 1000 25 \
  --no-plots
```

## Troubleshooting

1. **DLL Loading Errors**: Ensure you're running on Windows with .NET Framework and pythonnet installed
2. **Missing Material Files**: Check that all materials referenced in the stack XML exist in the Materials directory
3. **Memory Issues**: Reduce parameter ranges for large grids; consider using npy format instead of xlsx
4. **Path Issues**: Use absolute paths if relative paths cause problems
5. **Plot Generation Issues**: Install plotly with `pip install plotly`
6. **YAML Configuration Errors**: Validate YAML syntax and check required sections are present

## Performance Notes

- Generation time scales as: `lipid_steps × aqueous_steps × rough_steps × wavelength_count`
- Typical generation rate: ~1000-10000 spectra per minute
- Memory usage scales with grid size; large grids may require 8GB+ RAM
- Use NPY format for large datasets (much faster than Excel)
