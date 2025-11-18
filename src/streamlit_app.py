"""
Interactive Streamlit + Plotly app for exploring tear film reflectance spectra.

Features:
- Sliders for lipid thickness, aqueous thickness, and roughness
- Measurement spectra overlay from specified directory
- Side-by-side comparison of measured vs theoretical spectra
- Fit metrics calculation
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Optional
import itertools
import random
import os

from analysis import (
    load_measurement_spectrum,
    detrend_signal,
    detect_peaks,
    detect_valleys,
    prepare_measurement,
    prepare_theoretical_spectrum,
    peak_count_score,
    peak_delta_score,
    phase_overlap_score,
    composite_score,
)

from tear_film_generator import (
    load_config,
    validate_config,
    make_single_spectrum_calculator,
    PROJECT_ROOT,
    get_project_path,
)


def clamp_to_step(value: float, min_val: float, step: float) -> float:
    """Snap a value to the nearest step from min_val."""
    return min_val + round((value - min_val) / step) * step


def detrend_dataframe(df: pd.DataFrame, cutoff_frequency: float, filter_order: int) -> pd.DataFrame:
    """Apply shared detrending routine and attach the result to the dataframe."""

    df = df.sort_values(by="wavelength").reset_index(drop=True)
    detrended = detrend_signal(
        df["wavelength"].to_numpy(),
        df["reflectance"].to_numpy(),
        cutoff_frequency,
        filter_order,
    )
    df_result = df.copy()
    df_result["detrended"] = detrended
    return df_result


def detect_peaks_df(df: pd.DataFrame, column: str, prominence: float, height: Optional[float] = None) -> pd.DataFrame:
    """Convenience wrapper using the shared peak detector."""

    peaks = detect_peaks(
        df["wavelength"].to_numpy(),
        df[column].to_numpy(),
        prominence=prominence,
        height=height,
    )
    result = pd.DataFrame({
        "wavelength": peaks["wavelength"],
        column: peaks["amplitude"],
        "peak_prominence": peaks["prominence"],
    })
    return result


def detect_valleys_df(df: pd.DataFrame, column: str, prominence: float) -> pd.DataFrame:
    """Convenience wrapper for detecting valleys."""

    valleys = detect_valleys(
        df["wavelength"].to_numpy(),
        df[column].to_numpy(),
        prominence=prominence,
    )
    result = pd.DataFrame({
        "wavelength": valleys["wavelength"],
        column: valleys["amplitude"],
        "valley_prominence": valleys["prominence"],
    })
    return result


def load_measurement_files(measurements_dir: pathlib.Path, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load measurement spectra using the shared loader.
    
    Searches both the configured measurements directory and exploration/sample_data/
    to support Silas's data structure with good_fit/bad_fit folders.
    """

    measurements: Dict[str, pd.DataFrame] = {}
    meas_config = config.get("measurements", {})
    file_pattern = meas_config.get("file_pattern", "*.txt")
    
    # Search in both configured directory and exploration/sample_data/
    search_dirs = [measurements_dir]
    exploration_dir = PROJECT_ROOT / "exploration" / "sample_data"
    if exploration_dir.exists():
        search_dirs.append(exploration_dir)
    
    all_file_path_objs = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Normalize the base directory path for Windows UNC paths
        search_dir_normalized = pathlib.Path(os.path.normpath(str(search_dir.resolve())))
        file_path_objs = [
            p for p in search_dir_normalized.rglob(file_pattern)
            if p.is_file() and not p.name.endswith("_BestFit.txt")  # Skip BestFit files
        ]
        all_file_path_objs.extend(file_path_objs)

    if not all_file_path_objs:
        st.warning(f"No measurement files found matching pattern: {file_pattern}")
        return measurements

    for file_path_obj in sorted(all_file_path_objs):
        try:
            # Normalize path for Windows UNC paths (handle backslash issues)
            normalized_str = os.path.normpath(str(file_path_obj))
            file_path_obj = pathlib.Path(normalized_str)
            
            # Ensure path exists
            if not file_path_obj.exists():
                st.warning(f"File does not exist: {file_path_obj}")
                continue
            
            # Determine which base directory this file is relative to
            if exploration_dir.exists() and str(file_path_obj).startswith(str(exploration_dir.resolve())):
                base_dir = exploration_dir
            else:
                base_dir = measurements_dir
            
            # Include subdirectory path in key (e.g., "good_fit/(Run)spectra_09-46-14-250")
            base_dir_normalized = pathlib.Path(os.path.normpath(str(base_dir.resolve())))
            rel_path = file_path_obj.relative_to(base_dir_normalized)
            file_name = str(rel_path.with_suffix(""))  # Remove .txt extension
            meas_df = load_measurement_spectrum(file_path_obj, meas_config)
            if not meas_df.empty:
                measurements[file_name] = meas_df
        except Exception as exc:  # pragma: no cover - UI warning path
            st.warning(f"Error loading {file_path_obj}: {exc}")

    return measurements


def generate_parameter_values(cfg: Dict[str, Any], stride: int) -> np.ndarray:
    stride = max(1, stride)
    min_val = float(cfg["min"])
    max_val = float(cfg["max"])
    step = float(cfg["step"]) * stride
    if step <= 0:
        return np.array([min_val, max_val], dtype=float)

    values = np.arange(min_val, max_val + step * 0.5, step, dtype=float)
    if values.size == 0:
        values = np.array([min_val], dtype=float)

    if values[-1] < max_val - 1e-9:
        values = np.append(values, max_val)
    else:
        values[-1] = max_val
    return values


def score_candidate(
    measurement_features,
    theoretical_features,
    metrics_cfg: Dict[str, Any],
):
    peak_count_cfg = metrics_cfg.get("peak_count", {})
    peak_delta_cfg = metrics_cfg.get("peak_delta", {})
    weights = metrics_cfg.get("composite", {}).get("weights", {})

    count_result = peak_count_score(
        measurement_features,
        theoretical_features,
        tolerance_nm=float(peak_count_cfg.get("wavelength_tolerance_nm", 5.0)),
    )
    delta_result = peak_delta_score(
        measurement_features,
        theoretical_features,
        tolerance_nm=float(peak_delta_cfg.get("tolerance_nm", 5.0)),
        tau_nm=float(peak_delta_cfg.get("tau_nm", 15.0)),
        penalty_unpaired=float(peak_delta_cfg.get("penalty_unpaired", 0.05)),
    )
    phase_result = phase_overlap_score(measurement_features, theoretical_features)

    component_scores = {
        "peak_count": count_result.score,
        "peak_delta": delta_result.score,
        "phase_overlap": phase_result.score,
    }
    composite = composite_score(component_scores, weights)
    component_scores["composite"] = composite

    diagnostics = {
        "peak_count": count_result.diagnostics,
        "peak_delta": delta_result.diagnostics,
        "phase_overlap": phase_result.diagnostics,
    }
    return component_scores, diagnostics


def run_inline_grid_search(
    single_spectrum,
    wavelengths: np.ndarray,
    measurement_features,
    analysis_cfg: Dict[str, Any],
    metrics_cfg: Dict[str, Any],
    lipid_vals: np.ndarray,
    aqueous_vals: np.ndarray,
    rough_vals: np.ndarray,
    max_results: Optional[int],
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    evaluated = 0
    
    # Calculate total search space
    total_combinations = len(lipid_vals) * len(aqueous_vals) * len(rough_vals)
    
    # Use random sampling if max_results is set and covers less than 10% of search space
    # This prevents bias toward lower parameter values
    use_random_sampling = max_results is not None and max_results > 0 and max_results < total_combinations * 0.1
    
    if use_random_sampling:
        # Generate all combinations and randomly sample
        all_combinations = list(itertools.product(lipid_vals, aqueous_vals, rough_vals))
        # Use fixed seed for reproducibility
        random.seed(42)
        sampled_combinations = random.sample(all_combinations, min(max_results, len(all_combinations)))
        
        for lipid, aqueous, rough in sampled_combinations:
            spectrum = single_spectrum(float(lipid), float(aqueous), float(rough))
            # Validate spectrum is not flat/zero
            if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                # Skip invalid spectra
                continue
            # Check if spectrum has variation (before detrending)
            spectrum_std = np.std(spectrum)
            if spectrum_std < 1e-6:
                # Skip flat spectra
                continue
            theoretical = prepare_theoretical_spectrum(
                wavelengths,
                spectrum,
                measurement_features,
                analysis_cfg,
            )
            scores, diagnostics = score_candidate(
                measurement_features,
                theoretical,
                metrics_cfg,
            )
            record = {
                "lipid_nm": float(lipid),
                "aqueous_nm": float(aqueous),
                "roughness_A": float(rough),
            }
            for key, value in scores.items():
                record[f"score_{key}"] = float(value)
            for metric, diag in diagnostics.items():
                for diag_key, diag_val in diag.items():
                    record[f"{metric}_{diag_key}"] = float(diag_val)
            records.append(record)
            evaluated += 1
    else:
        # Use systematic grid search (original nested loop approach)
        for lipid in lipid_vals:
            for aqueous in aqueous_vals:
                for rough in rough_vals:
                    if max_results is not None and evaluated >= max_results:
                        break
                    spectrum = single_spectrum(float(lipid), float(aqueous), float(rough))
                    # Validate spectrum is not flat/zero
                    if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                        # Skip invalid spectra
                        continue
                    # Check if spectrum has variation (before detrending)
                    spectrum_std = np.std(spectrum)
                    if spectrum_std < 1e-6:
                        # Skip flat spectra
                        continue
                    theoretical = prepare_theoretical_spectrum(
                        wavelengths,
                        spectrum,
                        measurement_features,
                        analysis_cfg,
                    )
                    scores, diagnostics = score_candidate(
                        measurement_features,
                        theoretical,
                        metrics_cfg,
                    )
                    record = {
                        "lipid_nm": float(lipid),
                        "aqueous_nm": float(aqueous),
                        "roughness_A": float(rough),
                    }
                    for key, value in scores.items():
                        record[f"score_{key}"] = float(value)
                    for metric, diag in diagnostics.items():
                        for diag_key, diag_val in diag.items():
                            record[f"{metric}_{diag_key}"] = float(diag_val)
                    records.append(record)
                    evaluated += 1
                if max_results is not None and evaluated >= max_results:
                    break
            if max_results is not None and evaluated >= max_results:
                break

    if not records:
        return pd.DataFrame(), 0

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values("score_composite", ascending=False).reset_index(drop=True)
    return results_df, evaluated


def interpolate_measurement_to_theoretical(measured_df: pd.DataFrame, theoretical_wavelengths: np.ndarray) -> np.ndarray:
    """Interpolate measured spectrum to match theoretical wavelength grid."""
    return np.interp(theoretical_wavelengths, measured_df['wavelength'], measured_df['reflectance'])


def calculate_fit_metrics(measured: np.ndarray, theoretical: np.ndarray) -> Dict[str, float]:
    """Calculate goodness-of-fit metrics between measured and theoretical spectra."""
    # R-squared
    ss_res = np.sum((measured - theoretical) ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((measured - theoretical) ** 2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(measured - theoretical))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((measured - theoretical) / measured)) * 100 if np.mean(measured) != 0 else 0
    
    return {
        'R²': r_squared,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape
    }


def create_comparison_plot(theoretical_wl: np.ndarray, theoretical_spec: np.ndarray,
                          measured_df: pd.DataFrame, lipid_val: float, aqueous_val: float, 
                          rough_val: float, config: Dict[str, Any], selected_file: str) -> go.Figure:
    """Create a plot comparing theoretical and measured spectra."""
    plot_config = config.get('plotting', {})
    style = plot_config.get('plot_style', {})
    
    fig = go.Figure()
    
    # Add measured spectrum
    fig.add_trace(go.Scatter(
        x=measured_df['wavelength'],
        y=measured_df['reflectance'],
        mode='lines',
        name=f'Measured ({selected_file})',
        line=dict(
            color=style.get('measured_color', 'red'),
            width=style.get('line_width', 2)
        ),
        hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>Measured<extra></extra>'
    ))
    
    # Add theoretical spectrum
    fig.add_trace(go.Scatter(
        x=theoretical_wl,
        y=theoretical_spec,
        mode='lines',
        name=f'Theoretical (L={lipid_val:.0f}, A={aqueous_val:.0f}, R={rough_val:.0f})',
        line=dict(
            color=style.get('theoretical_color', 'blue'),
            width=style.get('line_width', 2)
        ),
        hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>Theoretical<extra></extra>'
    ))
    
    fig.update_layout(
        title="Measured vs Theoretical Reflectance Spectra",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        hovermode="x unified",
        template="plotly_white",
        width=style.get('width', 1000),
        height=style.get('height', 600),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def main():
    config = load_config(PROJECT_ROOT / "config.yaml")
    if not validate_config(config):
        st.error("Invalid configuration. See terminal for details.")
        st.stop()

    analysis_cfg: Dict[str, Any] = config.get("analysis", {})
    metrics_cfg: Dict[str, Any] = analysis_cfg.get("metrics", {})
    ui_cfg: Dict[str, Any] = config.get("ui", {})
    st.set_page_config(
        page_title=ui_cfg.get("page_title", "Tear Film Spectra Explorer"), 
        layout="wide"
    )

    # Load theoretical spectrum calculator
    single_spectrum, wavelengths = make_single_spectrum_calculator(config)

    # Load measurement files
    measurements_enabled = config.get('measurements', {}).get('enabled', False)
    measurements = {}
    
    if measurements_enabled:
        measurements_dir = get_project_path(config['paths']['measurements'])
        measurements = load_measurement_files(measurements_dir, config)

    params = config["parameters"]
    lipid_cfg = params["lipid"]
    aqueous_cfg = params["aqueous"]
    rough_cfg = params["roughness"]

    # Defaults: use configured defaults if provided, or midpoints snapped to step
    defaults = ui_cfg.get("default_values", {})
    def midpoint(cfg):
        return clamp_to_step((cfg["min"] + cfg["max"]) / 2, cfg["min"], cfg["step"])

    default_lipid = defaults.get("lipid", midpoint(lipid_cfg))
    default_aqueous = defaults.get("aqueous", midpoint(aqueous_cfg))
    default_rough = defaults.get("roughness", midpoint(rough_cfg))

    # Main content
    st.markdown("# Tear Film Spectra Explorer")
    st.markdown("Adjust layer properties to view theoretical reflectance spectrum and compare with measurements.")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Spectrum Comparison",
        "📈 Detrended Analysis",
        "⚙️ Parameters",
        "🔍 Grid Search",
    ])
    
    # Update sliders if a grid-search selection was applied in the previous run
    pending_update = st.session_state.pop("pending_slider_update", None)
    if pending_update:
        for slider_key, slider_value in pending_update.items():
            st.session_state[slider_key] = slider_value

    # Sidebar controls
    st.sidebar.markdown("## Layer Parameters")
    
    lipid_val = st.sidebar.slider(
        "Lipid thickness (nm)",
        min_value=float(lipid_cfg["min"]),
        max_value=float(lipid_cfg["max"]),
        value=float(default_lipid),
        step=float(lipid_cfg["step"]),
        format="%.0f",
        key="lipid_slider",
    )

    aqueous_val = st.sidebar.slider(
        "Aqueous thickness (nm)",
        min_value=float(aqueous_cfg["min"]),
        max_value=float(aqueous_cfg["max"]),
        value=float(default_aqueous),
        step=float(aqueous_cfg["step"]),
        format="%.0f",
        key="aqueous_slider",
    )

    rough_val = st.sidebar.slider(
        "Mucus roughness (Å)",
        min_value=float(rough_cfg["min"]),
        max_value=float(rough_cfg["max"]),
        value=float(default_rough),
        step=float(rough_cfg["step"]),
        format="%.0f",
        key="rough_slider",
    )

    # Measurement file selection
    selected_file = None
    if measurements_enabled and measurements:
        st.sidebar.markdown("## Measurement Data")
        measurement_files = list(measurements.keys())
        selected_file = st.sidebar.selectbox(
            "Select measurement file:",
            options=["None"] + measurement_files,
            index=1 if measurement_files else 0
        )
        
        if selected_file != "None":
            selected_measurement = measurements[selected_file]
            st.sidebar.write(f"**{selected_file}**")
            st.sidebar.write(f"Data points: {len(selected_measurement)}")
            st.sidebar.write(f"Wavelength range: {selected_measurement['wavelength'].min():.1f} - {selected_measurement['wavelength'].max():.1f} nm")

    # Detrending parameters
    st.sidebar.markdown("## Analysis Parameters")
    cutoff_freq = st.sidebar.slider("Detrending Cutoff Frequency", 0.001, 0.1, 0.01, 0.001, format="%.3f")
    peak_prominence = st.sidebar.slider("Peak Prominence", 0.001, 0.05, 0.005, 0.001, format="%.3f")

    # Compute theoretical spectrum
    spectrum = single_spectrum(lipid_val, aqueous_val, rough_val)
    theoretical_df = pd.DataFrame({
        'wavelength': wavelengths,
        'reflectance': spectrum
    })

    # Tab 1: Spectrum Comparison
    with tab1:
        if measurements_enabled and selected_file and selected_file != "None":
            selected_measurement = measurements[selected_file]
            
            # Create comparison plot
            fig = create_comparison_plot(
                wavelengths, spectrum, selected_measurement,
                lipid_val, aqueous_val, rough_val, config, selected_file
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display fit metrics
            interpolated_measured = interpolate_measurement_to_theoretical(selected_measurement, wavelengths)
            metrics = calculate_fit_metrics(interpolated_measured, spectrum)
            
            st.markdown("## Goodness of Fit Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R²", f"{metrics['R²']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.6f}")
            with col3:
                st.metric("MAE", f"{metrics['MAE']:.6f}")
            with col4:
                st.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
            
            # Residuals plot
            residuals = interpolated_measured - spectrum
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=wavelengths,
                y=residuals,
                mode='lines',
                name='Residuals (Measured - Theoretical)',
                line=dict(color='green', width=2)
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            fig_residuals.update_layout(
                title="Residuals (Measured - Theoretical)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Residual Reflectance",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
            
        else:
            # Show only theoretical spectrum
            fig = go.Figure()
            fig.add_scatter(
                x=wavelengths,
                y=spectrum,
                mode="lines",
                name=f"Theoretical (L={lipid_val:.0f}, A={aqueous_val:.0f}, R={rough_val:.0f})",
                line=dict(color=config.get('plotting', {}).get('plot_style', {}).get('theoretical_color', 'blue'))
            )
            fig.update_layout(
                title="Theoretical Reflectance Spectrum",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Detrended Analysis
    with tab2:
        st.markdown("## Detrended Signal Analysis with Peak Detection")
        
        if measurements_enabled and selected_file and selected_file != "None":
            selected_measurement = measurements[selected_file]
            
            # Detrend both theoretical and measured signals
            filter_order = config.get("analysis", {}).get("detrending", {}).get("filter_order", 3)
            theoretical_detrended = detrend_dataframe(theoretical_df, cutoff_freq, filter_order)
            measured_detrended = detrend_dataframe(selected_measurement, cutoff_freq, filter_order)

            # Detect peaks in detrended signals
            theo_peaks = detect_peaks_df(theoretical_detrended, 'detrended', peak_prominence)
            meas_peaks = detect_peaks_df(measured_detrended, 'detrended', peak_prominence)

            # Detect valleys in detrended signals
            theo_valleys = detect_valleys_df(theoretical_detrended, 'detrended', peak_prominence)
            meas_valleys = detect_valleys_df(measured_detrended, 'detrended', peak_prominence)
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Original vs Detrended Signals", "Peak and Valley Detection"),
                vertical_spacing=0.1
            )
            
            # Top plot: Original vs Detrended
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'], 
                y=theoretical_detrended['reflectance'],
                mode='lines', name='Theoretical Original',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=measured_detrended['wavelength'], 
                y=measured_detrended['reflectance'],
                mode='lines', name='Measured Original',
                line=dict(color='red', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'], 
                y=theoretical_detrended['detrended'],
                mode='lines', name='Theoretical Detrended',
                line=dict(color='lightblue', width=2, dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=measured_detrended['wavelength'], 
                y=measured_detrended['detrended'],
                mode='lines', name='Measured Detrended',
                line=dict(color='lightcoral', width=2, dash='dash')
            ), row=1, col=1)
            
            # Bottom plot: Detrended with peaks and valleys
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'], 
                y=theoretical_detrended['detrended'],
                mode='lines', name='Theoretical Detrended',
                line=dict(color='blue', width=2), showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=measured_detrended['wavelength'], 
                y=measured_detrended['detrended'],
                mode='lines', name='Measured Detrended',
                line=dict(color='red', width=2), showlegend=False
            ), row=2, col=1)
            
            # Add peaks
            if len(theo_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_peaks['wavelength'], 
                    y=theo_peaks['detrended'],
                    mode='markers', name='Theoretical Peaks',
                    marker=dict(color='blue', size=8, symbol='triangle-up')
                ), row=2, col=1)
            
            if len(meas_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=meas_peaks['wavelength'], 
                    y=meas_peaks['detrended'],
                    mode='markers', name='Measured Peaks',
                    marker=dict(color='red', size=8, symbol='triangle-up')
                ), row=2, col=1)
            
            # Add valleys
            if len(theo_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_valleys['wavelength'], 
                    y=theo_valleys['detrended'],
                    mode='markers', name='Theoretical Valleys',
                    marker=dict(color='blue', size=8, symbol='triangle-down')
                ), row=2, col=1)
            
            if len(meas_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=meas_valleys['wavelength'], 
                    y=meas_valleys['detrended'],
                    mode='markers', name='Measured Valleys',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ), row=2, col=1)
            
            fig.update_xaxes(title_text="Wavelength (nm)")
            fig.update_yaxes(title_text="Reflectance", row=1, col=1)
            fig.update_yaxes(title_text="Detrended Reflectance", row=2, col=1)
            fig.update_layout(height=800, template="plotly_white")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak analysis summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Theoretical Signal Analysis")
                st.write(f"**Peaks detected:** {len(theo_peaks)}")
                if len(theo_peaks) > 0:
                    st.write("Peak wavelengths:", [f"{w:.1f}nm" for w in theo_peaks['wavelength'].head(5)])
                st.write(f"**Valleys detected:** {len(theo_valleys)}")
                if len(theo_valleys) > 0:
                    st.write("Valley wavelengths:", [f"{w:.1f}nm" for w in theo_valleys['wavelength'].head(5)])
            
            with col2:
                st.markdown("### Measured Signal Analysis")
                st.write(f"**Peaks detected:** {len(meas_peaks)}")
                if len(meas_peaks) > 0:
                    st.write("Peak wavelengths:", [f"{w:.1f}nm" for w in meas_peaks['wavelength'].head(5)])
                st.write(f"**Valleys detected:** {len(meas_valleys)}")
                if len(meas_valleys) > 0:
                    st.write("Valley wavelengths:", [f"{w:.1f}nm" for w in meas_valleys['wavelength'].head(5)])
            
        else:
            # Show only theoretical detrended
            filter_order = config.get("analysis", {}).get("detrending", {}).get("filter_order", 3)
            theoretical_detrended = detrend_dataframe(theoretical_df, cutoff_freq, filter_order)
            theo_peaks = detect_peaks_df(theoretical_detrended, 'detrended', peak_prominence)
            theo_valleys = detect_valleys_df(theoretical_detrended, 'detrended', peak_prominence)
            
            fig = go.Figure()
            
            # Original signal
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'],
                y=theoretical_detrended['reflectance'],
                mode='lines', name='Original',
                line=dict(color='blue', width=2)
            ))
            
            # Detrended signal
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'],
                y=theoretical_detrended['detrended'],
                mode='lines', name='Detrended',
                line=dict(color='lightblue', width=2, dash='dash')
            ))
            
            # Peaks
            if len(theo_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_peaks['wavelength'],
                    y=theo_peaks['detrended'],
                    mode='markers', name='Peaks',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ))
            
            # Valleys
            if len(theo_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_valleys['wavelength'],
                    y=theo_valleys['detrended'],
                    mode='markers', name='Valleys',
                    marker=dict(color='orange', size=8, symbol='triangle-down')
                ))
            
            fig.update_layout(
                title="Theoretical Signal: Original vs Detrended with Peak/Valley Detection",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Analysis Summary")
            st.write(f"**Peaks detected:** {len(theo_peaks)}")
            if len(theo_peaks) > 0:
                st.write("Peak wavelengths:", [f"{w:.1f}nm" for w in theo_peaks['wavelength']])
            st.write(f"**Valleys detected:** {len(theo_valleys)}")
            if len(theo_valleys) > 0:
                st.write("Valley wavelengths:", [f"{w:.1f}nm" for w in theo_valleys['wavelength']])

    # Tab 3: Parameters
    with tab3:
        st.markdown("## Current Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Layer Properties")
            st.json({
                "lipid_thickness_nm": float(lipid_val),
                "aqueous_thickness_nm": float(aqueous_val),
                "mucus_roughness_angstrom": float(rough_val),
                "mucus_thickness_nm": config.get('fixed', {}).get('mucus_thickness', 500),
            })
        
        with col2:
            st.markdown("### Analysis Settings")
            st.json({
                "detrending_cutoff_frequency": float(cutoff_freq),
                "peak_detection_prominence": float(peak_prominence),
                "selected_measurement": selected_file if selected_file and selected_file != "None" else None
            })

        # Measurement data info
        if measurements_enabled:
            st.markdown("### Available Measurement Files")
            if measurements:
                for filename, df in measurements.items():
                    st.write(f"**{filename}**: {len(df)} data points, λ = {df['wavelength'].min():.1f}-{df['wavelength'].max():.1f} nm")
            else:
                st.info(f"No measurement files found in: {get_project_path(config['paths']['measurements'])}")
        else:
            st.info("Measurement comparison is disabled. Enable in config.yaml under 'measurements.enabled' to compare with experimental data.")

    with tab4:
        st.markdown("## Grid Search Ranking")
        if not (measurements_enabled and selected_file and selected_file != "None"):
            st.info("Select a measurement spectrum to run the grid search.")
        else:
            selected_measurement = measurements[selected_file]
            controls = st.columns(3)
            top_k_display = int(
                controls[0].number_input("Top results", min_value=5, max_value=100, value=10, step=5)
            )
            stride = int(
                controls[1].number_input("Stride multiplier", min_value=1, max_value=10, value=1, step=1)
            )
            max_results_input = int(
                controls[2].number_input("Max spectra", min_value=0, value=500, step=50)
            )
            max_results = None if max_results_input == 0 else max_results_input

            run_pressed = st.button("Run Grid Search", key="run_grid_search_button")
            cache_key = f"grid_search_{selected_file}"
            if run_pressed:
                with st.spinner("Scoring theoretical spectra..."):
                    measurement_features = prepare_measurement(selected_measurement, analysis_cfg)
                    lipid_vals = generate_parameter_values(config["parameters"]["lipid"], stride)
                    aqueous_vals = generate_parameter_values(config["parameters"]["aqueous"], stride)
                    rough_vals = generate_parameter_values(config["parameters"]["roughness"], stride)
                    results_df, evaluated = run_inline_grid_search(
                        single_spectrum,
                        wavelengths,
                        measurement_features,
                        analysis_cfg,
                        metrics_cfg,
                        lipid_vals,
                        aqueous_vals,
                        rough_vals,
                        max_results,
                    )
                    st.session_state[cache_key] = {
                        "results": results_df,
                        "evaluated": evaluated,
                        "stride": stride,
                        "max_results": max_results_input,
                        "lipid_vals": lipid_vals,
                        "aqueous_vals": aqueous_vals,
                        "rough_vals": rough_vals,
                    }

            cache_entry = st.session_state.get(cache_key)
            if cache_entry:
                results_df = cache_entry["results"]
                evaluated = cache_entry.get("evaluated", len(results_df))
                if results_df.empty:
                    st.warning("No spectra were evaluated with the current settings.")
                else:
                    st.caption(
                        f"Evaluated {evaluated} spectra (stride ×{cache_entry.get('stride', stride)})."
                    )
                    # Show score statistics for debugging
                    if "score_composite" in results_df.columns:
                        score_min = results_df["score_composite"].min()
                        score_max = results_df["score_composite"].max()
                        score_mean = results_df["score_composite"].mean()
                        st.caption(f"Composite scores: min={score_min:.4f}, max={score_max:.4f}, mean={score_mean:.4f}")
                    
                    # Show peak detection diagnostics
                    if "peak_count_measurement_peaks" in results_df.columns:
                        meas_peaks = results_df["peak_count_measurement_peaks"].iloc[0] if len(results_df) > 0 else 0
                        theo_peaks_col = "peak_count_theoretical_peaks"
                        if theo_peaks_col in results_df.columns:
                            theo_peaks_avg = results_df[theo_peaks_col].mean()
                            theo_peaks_max = results_df[theo_peaks_col].max()
                            st.caption(f"Peak detection: measurement={int(meas_peaks)} peaks, theoretical avg={theo_peaks_avg:.1f}, max={int(theo_peaks_max)}")
                            if theo_peaks_max == 0:
                                st.warning("⚠️ No peaks detected in theoretical spectra! Try lowering the prominence threshold in Analysis Parameters.")
                    
                    # Show parameter range diagnostics
                    if "lipid_nm" in results_df.columns:
                        lipid_min = results_df["lipid_nm"].min()
                        lipid_max = results_df["lipid_nm"].max()
                        lipid_unique = results_df["lipid_nm"].nunique()
                        aqueous_min = results_df["aqueous_nm"].min()
                        aqueous_max = results_df["aqueous_nm"].max()
                        aqueous_unique = results_df["aqueous_nm"].nunique()
                        rough_min = results_df["roughness_A"].min()
                        rough_max = results_df["roughness_A"].max()
                        rough_unique = results_df["roughness_A"].nunique()
                        st.caption(f"Parameter ranges explored: lipid={lipid_min:.0f}-{lipid_max:.0f} ({lipid_unique} values), "
                                 f"aqueous={aqueous_min:.0f}-{aqueous_max:.0f} ({aqueous_unique} values), "
                                 f"roughness={rough_min:.0f}-{rough_max:.0f} ({rough_unique} values)")
                        
                        # Warn if search space is too large
                        cached_lipid_vals = cache_entry.get("lipid_vals")
                        cached_aqueous_vals = cache_entry.get("aqueous_vals")
                        cached_rough_vals = cache_entry.get("rough_vals")
                        if cached_lipid_vals is not None and cached_aqueous_vals is not None and cached_rough_vals is not None:
                            total_combinations = len(cached_lipid_vals) * len(cached_aqueous_vals) * len(cached_rough_vals)
                            max_results_cached = cache_entry.get("max_results")
                            if max_results_cached is not None and max_results_cached > 0 and max_results_cached < total_combinations * 0.1:
                                coverage = 100 * evaluated / total_combinations
                                st.warning(f"⚠️ Only {coverage:.1f}% of parameter space explored ({evaluated}/{total_combinations} combinations). "
                                         f"Grid search may be biased toward lower parameter values. "
                                         f"Consider increasing 'Max spectra' or using a larger 'Stride multiplier'.")
                    
                    display_df = results_df.head(top_k_display)
                    st.dataframe(display_df, use_container_width=True)
                    options = list(display_df.index)
                    selection = st.selectbox(
                        "Select a candidate to apply", options=options, format_func=lambda idx: f"Rank {idx + 1}"
                    )
                    if st.button("Apply Selection", key="apply_grid_selection"):
                        row = display_df.loc[selection]
                        st.session_state["pending_slider_update"] = {
                            "lipid_slider": float(row["lipid_nm"]),
                            "aqueous_slider": float(row["aqueous_nm"]),
                            "rough_slider": float(row["roughness_A"]),
                        }
                        st.rerun()
            else:
                st.info("Run the grid search to see ranked candidates.")


if __name__ == "__main__":
    main()


