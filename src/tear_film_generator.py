# -*- coding: utf-8 -*-
"""
Generate a 4-D grid (lipid × aqueous × rough × λ) of theoretical tear-film
reflectance spectra using AdOM's SpAnalizer.dll.

Reorganized version with proper folder structure:
- configs/: XML configuration files
- data/: Materials and wavelength data
- src/: Source code and DLL
- outputs/: Generated spectra and results

CLI flags:
  --lipid   MIN MAX STEP      nm   (e.g. 0 256 1  → 0-255 nm)
  --aqueous MIN MAX STEP      nm   (e.g. 1000 5001 10 → 1000-5000 nm)
  --rough   MIN MAX STEP      Å    (e.g. 1000 2701 100)
  --wavelengths CSV-file      (vector λ)
  --out {npy|xlsx}

© 2025
"""

import argparse, json, os, pathlib, tempfile, uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np, pandas as pd
import yaml
from xml.etree import ElementTree as ET

try:
    import clr
    CLR_AVAILABLE = True
except ImportError:
    CLR_AVAILABLE = False
    print("Warning: pythonnet (clr) not available. Only demo mode will work.")
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

# Get the project root directory (parent of src/)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

def get_project_path(relative_path: str) -> pathlib.Path:
    """Convert relative path from project root to absolute path."""
    return PROJECT_ROOT / relative_path

def load_config(config_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply preset if specified
    if 'active_preset' in config and config['active_preset']:
        preset_name = config['active_preset']
        if preset_name in config.get('presets', {}):
            preset = config['presets'][preset_name]
            # Override parameters with preset values
            config['parameters'] = preset
            print(f"Using preset: {preset_name}")
        else:
            print(f"Warning: Preset '{preset_name}' not found, using manual parameters")
    
    return config

def create_spectral_plots(grid: np.ndarray, wl: np.ndarray, 
                         lipid_vals: np.ndarray, aqueous_vals: np.ndarray, 
                         rough_vals: np.ndarray, config: Dict[str, Any],
                         output_dir: pathlib.Path) -> None:
    """Create interactive plots of the generated spectra."""
    
    if not PLOTLY_AVAILABLE:
        print("Skipping plot generation - Plotly not available")
        return
    
    plot_config = config.get('plotting', {})
    style = plot_config.get('plot_style', {})
    
    # Determine which spectra to plot
    spectra_to_plot = plot_config.get('spectra_to_plot', 'sample')
    
    if spectra_to_plot == 'sample':
        num_samples = plot_config.get('num_samples', 6)
        # Sample evenly across parameter space
        indices = []
        for i in range(min(num_samples, len(lipid_vals))):
            for j in range(min(num_samples // len(lipid_vals) + 1, len(aqueous_vals))):
                for k in range(min(num_samples // (len(lipid_vals) * len(aqueous_vals)) + 1, len(rough_vals))):
                    if len(indices) < num_samples:
                        indices.append((i, j, k))
    elif spectra_to_plot == 'all':
        indices = [(i, j, k) for i in range(len(lipid_vals)) 
                  for j in range(len(aqueous_vals)) 
                  for k in range(len(rough_vals))]
    else:
        indices = spectra_to_plot
    
    # Create main spectral plot
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for idx, (i, j, k) in enumerate(indices):
        spectrum = grid[i, j, k, :]
        l_val = lipid_vals[i]
        a_val = aqueous_vals[j]
        r_val = rough_vals[k]
        
        fig.add_trace(go.Scatter(
            x=wl, 
            y=spectrum,
            mode='lines',
            name=f'L={l_val:.0f}nm, A={a_val:.0f}nm, R={r_val:.0f}Å',
            line=dict(color=colors[idx % len(colors)], 
                     width=style.get('line_width', 2)),
            hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>%{fullData.name}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Tear Film Theoretical Reflectance Spectra',
            font=dict(size=style.get('title_size', 16))
        ),
        xaxis=dict(
            title='Wavelength (nm)',
            title_font=dict(size=style.get('axis_label_size', 14))
        ),
        yaxis=dict(
            title='Reflectance',
            title_font=dict(size=style.get('axis_label_size', 14))
        ),
        width=style.get('width', 1000),
        height=style.get('height', 600),
        hovermode='x unified'
    )
    
    # Save main plot
    plot_format = config.get('output', {}).get('plot_format', 'html')
    if plot_format == 'html':
        fig.write_html(output_dir / 'spectra_plot.html')
        print(f"✓ Saved: {output_dir / 'spectra_plot.html'}")
    elif plot_format == 'png':
        fig.write_image(output_dir / 'spectra_plot.png')
        print(f"✓ Saved: {output_dir / 'spectra_plot.png'}")
    elif plot_format == 'pdf':
        fig.write_image(output_dir / 'spectra_plot.pdf')
        print(f"✓ Saved: {output_dir / 'spectra_plot.pdf'}")
    
    # Create parameter sweep visualization
    create_parameter_sweep_plots(grid, wl, lipid_vals, aqueous_vals, rough_vals, 
                                config, output_dir)

def create_parameter_sweep_plots(grid: np.ndarray, wl: np.ndarray,
                                lipid_vals: np.ndarray, aqueous_vals: np.ndarray,
                                rough_vals: np.ndarray, config: Dict[str, Any],
                                output_dir: pathlib.Path) -> None:
    """Create heatmaps showing parameter dependencies."""
    
    if not PLOTLY_AVAILABLE:
        return
    
    # Select a few key wavelengths for analysis
    key_wavelengths = [650, 750, 850, 950]
    wl_indices = [np.argmin(np.abs(wl - target_wl)) for target_wl in key_wavelengths]
    actual_wavelengths = [wl[idx] for idx in wl_indices]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{actual_wl:.1f} nm' for actual_wl in actual_wavelengths],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # For each wavelength, create a heatmap of reflectance vs lipid and aqueous thickness
    # (averaging over roughness values)
    for plot_idx, (wl_idx, actual_wl) in enumerate(zip(wl_indices, actual_wavelengths)):
        row = plot_idx // 2 + 1
        col = plot_idx % 2 + 1
        
        # Average over roughness dimension
        reflectance_map = np.mean(grid[:, :, :, wl_idx], axis=2)
        
        fig.add_trace(
            go.Heatmap(
                x=aqueous_vals,
                y=lipid_vals,
                z=reflectance_map,
                colorscale='Viridis',
                showscale=(plot_idx == 0),  # Only show colorbar for first plot
                hovertemplate='Aqueous: %{x:.0f}nm<br>Lipid: %{y:.0f}nm<br>Reflectance: %{z:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=dict(
            text='Reflectance vs Layer Thicknesses (averaged over roughness)',
            font=dict(size=16)
        ),
        height=800
    )
    
    # Update axis labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Aqueous Thickness (nm)", row=i, col=j)
            fig.update_yaxes(title_text="Lipid Thickness (nm)", row=i, col=j)
    
    # Save parameter sweep plot
    plot_format = config.get('output', {}).get('plot_format', 'html')
    if plot_format == 'html':
        fig.write_html(output_dir / 'parameter_sweep.html')
        print(f"✓ Saved: {output_dir / 'parameter_sweep.html'}")
    elif plot_format == 'png':
        fig.write_image(output_dir / 'parameter_sweep.png', width=1200, height=800)
        print(f"✓ Saved: {output_dir / 'parameter_sweep.png'}")
    elif plot_format == 'pdf':
        fig.write_image(output_dir / 'parameter_sweep.pdf', width=1200, height=800)
        print(f"✓ Saved: {output_dir / 'parameter_sweep.pdf'}")

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters against XML constraints."""
    
    # Check required sections
    required_sections = ['parameters', 'paths', 'output']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in config")
            return False
    
    # Load XML constraints for validation
    try:
        from xml_constraint_parser import parse_algorithm_constraints, parse_stack_defaults, validate_config_against_constraints
        
        paths = config['paths']
        algorithm_path = get_project_path(paths['algorithm'])
        stack_path = get_project_path(paths['stack'])
        
        algorithm_constraints = parse_algorithm_constraints(algorithm_path)
        stack_defaults = parse_stack_defaults(stack_path)
        
        # Validate against XML constraints
        is_valid, errors = validate_config_against_constraints(config, algorithm_constraints, stack_defaults)
        
        if not is_valid:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            print(f"\nAlgorithm constraints:")
            for param, constraints in algorithm_constraints.items():
                print(f"  {param}: {constraints['min']}-{constraints['max']} (step: {constraints['step']})")
            return False
        
    except ImportError:
        print("Warning: XML constraint validation not available")
    except Exception as e:
        print(f"Warning: Could not validate against XML constraints: {e}")
    
    # Basic parameter validation
    params = config['parameters']
    for param_name in ['lipid', 'aqueous', 'roughness']:
        if param_name not in params:
            print(f"Error: Missing parameter '{param_name}' in config")
            return False
        
        param = params[param_name]
        if not all(key in param for key in ['min', 'max', 'step']):
            print(f"Error: Parameter '{param_name}' missing min/max/step values")
            return False
        
        if param['min'] >= param['max']:
            print(f"Error: Parameter '{param_name}' min >= max")
            return False
        
        if param['step'] <= 0:
            print(f"Error: Parameter '{param_name}' step <= 0")
            return False
    
    # Validate measurements directory if enabled
    if config.get('measurements', {}).get('enabled', False):
        measurements_path = get_project_path(config['paths']['measurements'])
        if not measurements_path.exists():
            print(f"Warning: Measurements directory not found: {measurements_path}")
    
    return True

def prepare_wavelengths_from_config(config: Dict[str, Any]) -> tuple[pathlib.Path, np.ndarray]:
    """Resolve wavelengths CSV path from config and return (path, vector)."""
    paths = config.get('paths', {})
    wl_config = config.get('wavelengths', {})
    if 'file' in wl_config:
        wavelengths_path = get_project_path(paths.get('wavelengths_dir', 'data/wavelengths')) / wl_config['file']
    else:
        wavelengths_path = get_project_path("data/wavelengths/Wavelengths813.csv")
    wl = pd.read_csv(wavelengths_path, header=None).iloc[:, 0].to_numpy(float)
    return wavelengths_path, wl

def make_single_spectrum_calculator(config: Dict[str, Any]):
    """Prepare a fast single-spectrum calculator and the wavelength vector.

    Returns a tuple: (calc_func, wavelengths)
    - calc_func(lipid_nm: float, aqueous_nm: float, rough_A: float) -> np.ndarray
    - wavelengths: np.ndarray
    """
    paths = config['paths']
    dll_path = get_project_path(paths['dll'])
    algorithm_path = get_project_path(paths['algorithm'])
    config_xml_path = get_project_path(paths['configuration'])
    materials_path = get_project_path(paths['materials'])
    base_stack_path = get_project_path(paths['stack'])

    # Prepare wavelength vector
    _, wl = prepare_wavelengths_from_config(config)

    # Prepare DLL-backed calculator and base stack once
    calc_func = make_calc_refl(dll_path, algorithm_path, materials_path, config_xml_path)
    base_stack = ET.parse(base_stack_path).getroot()
    mucus_nm_default = config.get('fixed', {}).get('mucus_thickness', 500)

    def single_spectrum(lipid_nm: float, aqueous_nm: float, rough_A: float) -> np.ndarray:
        xml = build_stack_xml(
            lipid_nm=lipid_nm,
            aqueous_nm=aqueous_nm,
            mucus_nm=mucus_nm_default,
            mucus_rough=rough_A,
            materials_dir=materials_path,
            base_stack=base_stack,
        )
        try:
            refl = np.asarray(calc_func(wl.tolist(), xml), dtype=np.float32)
        finally:
            try:
                os.remove(xml)
            except Exception:
                pass
        return refl

    return single_spectrum, wl

def build_stack_xml(lipid_nm: float, aqueous_nm: float, *, mucus_nm: float = 500,
                    mucus_rough: float, materials_dir: pathlib.Path,
                    base_stack: ET.Element) -> str:
    """Return path of a temp-stack XML with patched thicknesses / roughness."""
    stack = ET.fromstring(ET.tostring(base_stack))          # deep-copy
    for layer in stack.findall("Layer"):
        name = layer.findtext("Name")
        if   name == "Lipid":   layer.find("Height").text = f"{lipid_nm}"
        elif name == "Aqueous": layer.find("Height").text = f"{aqueous_nm}"
        elif name == "Mucus":
            layer.find("Height").text    = f"{mucus_nm}"
            layer.find("Roughness").text = f"{mucus_rough}"

        mat = layer.findtext("Material")
        csv = materials_dir / f"{mat}.csv"
        if not csv.is_file():
            raise FileNotFoundError(f"Missing material file: {csv}")

    tmp = pathlib.Path(tempfile.gettempdir())/f"stk_{uuid.uuid4().hex}.xml"
    ET.ElementTree(stack).write(tmp, encoding="utf-8", xml_declaration=True)
    return str(tmp)

def make_calc_refl(dll: pathlib.Path, algorithm_xml: pathlib.Path,
                   materials_dir: pathlib.Path, config_xml: pathlib.Path | None):
    """Return callable wavelengths,list -> R(λ)."""
    if not CLR_AVAILABLE:
        raise ImportError("pythonnet (clr) module required for DLL functionality. Install with: pip install pythonnet")
    
    clr.AddReference(str(dll))
    from SpAnalizer import (Serializer, SpAnalizeHelper,
                            PolarizationTypeEnum, AnalizerTypeEnum,
                            IParameter)
    from System.Collections.Generic import List

    ser     = Serializer()
    alg     = ser.ReadAlgorithmFromFile(AnalizerTypeEnum.Thickness, str(algorithm_xml))
    cfg     = ser.LoadConfigurationthFromFile(str(config_xml)) if config_xml else None
    helper  = SpAnalizeHelper()
    empty_p = List[IParameter]()

    def calc(wl: list[float], stack_xml: str):
        sa = helper.CreateSpectrumAnalizer(
                ser.ReadStackFromFile(stack_xml), alg, str(materials_dir), wl, cfg)
        return sa.CalcTheorSpectrum(empty_p, PolarizationTypeEnum.U)[0].get_Refl()

    return calc

def generate_grid(calc_func, wl: np.ndarray, lipid_vals, aqueous_vals, rough_vals,
                  *, materials_dir: pathlib.Path, base_stack_xml: pathlib.Path):
    """Generate the 4D reflectance grid."""
    base_stack = ET.parse(base_stack_xml).getroot()
    grid = np.empty((len(lipid_vals), len(aqueous_vals),
                     len(rough_vals), len(wl)), dtype=np.float32)

    total = len(lipid_vals)*len(aqueous_vals)*len(rough_vals)
    done  = 0
    for i,l in enumerate(lipid_vals):
        for j,a in enumerate(aqueous_vals):
            for k,r in enumerate(rough_vals):
                done += 1
                pct = 100*done/total
                print(f"\r{pct:6.2f}%  L:{l:>3} A:{a:>4} R:{int(r):>4}", end="", flush=True)
                xml = build_stack_xml(l, a, mucus_rough=r,
                                      materials_dir=materials_dir, base_stack=base_stack)
                grid[i,j,k,:] = calc_func(wl.tolist(), xml)
                os.remove(xml)
    print("\rDone – 100.00%"+ " "*40)
    return grid

def main():
    p = argparse.ArgumentParser("Tear-film grid generator")
    
    # Configuration file option
    p.add_argument("--config-file", type=pathlib.Path, 
                   help="YAML configuration file (default: config.yaml)")
    
    # Override options (will override config file values)
    p.add_argument("--dll", "--spanalizer", type=pathlib.Path, 
                   help="Path to SpAnalizer.dll")
    p.add_argument("--algorithm", type=pathlib.Path, 
                   help="Path to algorithm XML file")
    p.add_argument("--configuration", type=pathlib.Path, 
                   help="Path to configuration XML file")
    p.add_argument("--materials", type=pathlib.Path, 
                   help="Path to materials directory")
    p.add_argument("--wavelengths", type=pathlib.Path, 
                   help="Path to wavelengths CSV file")
    p.add_argument("--base_stack", type=pathlib.Path, 
                   help="Path to base stack XML file")
    
    # Parameter overrides
    p.add_argument("--lipid", nargs=3, type=float, 
                   help="Lipid thickness range: min max step")
    p.add_argument("--aqueous", nargs=3, type=float, 
                   help="Aqueous thickness range: min max step")
    p.add_argument("--rough", nargs=3, type=float, 
                   help="Roughness range: min max step")
    
    # Output overrides
    p.add_argument("--out", choices=("npy","xlsx"), 
                   help="Output format")
    p.add_argument("--out_dir", type=pathlib.Path, 
                   help="Output directory")
    p.add_argument("--no-plots", action="store_true", 
                   help="Skip plot generation")
    
    args = p.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    if not validate_config(config):
        return 1
    
    # Apply command line overrides
    paths = config['paths']
    
    # File paths (use config defaults, override with command line args)
    dll_path = get_project_path(args.dll or paths['dll'])
    algorithm_path = get_project_path(args.algorithm or paths['algorithm'])
    config_path = get_project_path(args.configuration or paths['configuration'])
    materials_path = get_project_path(args.materials or paths['materials'])
    base_stack_path = get_project_path(args.base_stack or paths['stack'])
    
    # Wavelengths
    if args.wavelengths:
        wavelengths_path = args.wavelengths
    else:
        wl_config = config.get('wavelengths', {})
        if 'file' in wl_config:
            wavelengths_path = get_project_path(paths['wavelengths_dir']) / wl_config['file']
        else:
            wavelengths_path = get_project_path("data/wavelengths/Wavelengths813.csv")
    
    # Parameters (use config defaults, override with command line args)
    params = config['parameters']
    lipid_range = args.lipid or [params['lipid']['min'], params['lipid']['max'], params['lipid']['step']]
    aqueous_range = args.aqueous or [params['aqueous']['min'], params['aqueous']['max'], params['aqueous']['step']]
    rough_range = args.rough or [params['roughness']['min'], params['roughness']['max'], params['roughness']['step']]
    
    # Output settings
    output_config = config.get('output', {})
    output_format = args.out or output_config.get('format', 'npy')
    output_dir = args.out_dir or get_project_path(output_config.get('base_dir', 'outputs'))
    include_plots = not args.no_plots and output_config.get('include_plots', True)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify all files exist
    required_files = [dll_path, algorithm_path, materials_path, wavelengths_path, base_stack_path, config_path]
    
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    print("=== Tear Film Theoretical Spectra Generator ===")
    print(f"Configuration: {config.get('active_preset', 'manual parameters')}")
    print(f"DLL: {dll_path}")
    print(f"Algorithm: {algorithm_path}")
    print(f"Config: {config_path}")
    print(f"Materials: {materials_path}")
    print(f"Wavelengths: {wavelengths_path}")
    print(f"Base Stack: {base_stack_path}")
    print(f"Output: {output_dir}")
    print()

    calc = make_calc_refl(dll_path, algorithm_path, materials_path, config_path)
    wl = pd.read_csv(wavelengths_path, header=None).iloc[:,0].to_numpy(float)
    lipid_vals = np.arange(*lipid_range, dtype=float)
    aqueous_vals = np.arange(*aqueous_range, dtype=float)
    rough_vals = np.arange(*rough_range, dtype=float)

    print(f"Parameter ranges:")
    print(f"  Lipid thickness: {lipid_vals[0]}-{lipid_vals[-1]} nm ({len(lipid_vals)} values)")
    print(f"  Aqueous thickness: {aqueous_vals[0]}-{aqueous_vals[-1]} nm ({len(aqueous_vals)} values)")
    print(f"  Roughness: {rough_vals[0]}-{rough_vals[-1]} Å ({len(rough_vals)} values)")
    print(f"  Wavelengths: {wl[0]:.1f}-{wl[-1]:.1f} nm ({len(wl)} values)")
    print(f"  Total spectra: {len(lipid_vals)} × {len(aqueous_vals)} × {len(rough_vals)} = {len(lipid_vals) * len(aqueous_vals) * len(rough_vals)}")
    print()

    grid = generate_grid(calc, wl, lipid_vals, aqueous_vals, rough_vals,
                         materials_dir=materials_path, base_stack_xml=base_stack_path)

    meta = dict(
        lipid_nm=lipid_vals.tolist(),
        aqueous_nm=aqueous_vals.tolist(),
        rough_A=rough_vals.tolist(),
        wavelengths_nm=wl.tolist(),
        config_used=config.get('active_preset', 'manual'),
        total_spectra=len(lipid_vals) * len(aqueous_vals) * len(rough_vals)
    )

    # Create timestamped output subdirectory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = output_dir / f"spectra_{timestamp}"
    output_subdir.mkdir(exist_ok=True)

    # Save data files
    if output_format == "npy":
        np.save(output_subdir / "grid.npy", grid)
        print(f"✓ Saved: {output_subdir / 'grid.npy'}")
    else:
        with pd.ExcelWriter(output_subdir / "grid.xlsx") as xl:
            for k, r in enumerate(rough_vals):
                for i, l in enumerate(lipid_vals):
                    sheet = f"L{l:.0f}_R{int(r)}"
                    df = pd.DataFrame(grid[i, :, k, :], index=aqueous_vals, columns=wl)
                    df.to_excel(xl, sheet_name=sheet[:31])
        print(f"✓ Saved: {output_subdir / 'grid.xlsx'}")

    # Save metadata
    with open(output_subdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved: {output_subdir / 'meta.json'}")

    # Save configuration used
    with open(output_subdir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Saved: {output_subdir / 'config_used.yaml'}")

    # Generate plots
    if include_plots:
        print("\nGenerating plots...")
        create_spectral_plots(grid, wl, lipid_vals, aqueous_vals, rough_vals, config, output_subdir)
    
    print(f"\n=== Generation Complete ===")
    print(f"Results saved in: {output_subdir}")
    if include_plots and PLOTLY_AVAILABLE:
        print(f"Open spectra_plot.html in a web browser to view interactive plots")

if __name__ == "__main__":
    main()
