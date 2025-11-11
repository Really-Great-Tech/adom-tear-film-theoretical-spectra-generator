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
import glob
import os
from scipy.signal import butter, filtfilt, find_peaks

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


def load_txt_file_enhanced(file_path):
    """
    Load spectral data from a given txt file, automatically detecting headers or metadata.
    Enhanced version based on your code.

    Args:
        file_path (str or Path): Path to the text file.

    Returns:
        pd.DataFrame: DataFrame with two columns ['wavelength', 'reflectance'].
    """
    data_started = False
    wavelengths = []
    intensities = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Detect the spectral data start (for non-best-fit files)
            if line.startswith('>>>>>Begin Spectral Data<<<<<'):
                data_started = True
                continue

            # Check if line contains two numerical columns
            parts = line.split()
            if len(parts) == 2:
                try:
                    wavelength, intensity = float(parts[0]), float(parts[1])
                    wavelengths.append(wavelength)
                    intensities.append(intensity)
                    data_started = True  # for BestFit files, data immediately starts
                except ValueError:
                    # Skip lines that don't have numerical data
                    if data_started:
                        # If data previously started but now hit non-numerical, break
                        break
                    else:
                        continue

    return pd.DataFrame({"wavelength": wavelengths, "reflectance": intensities})


def detrend_signal(df, cutoff_frequency=0.01, filter_order=3):
    """
    Detrends a signal using a high-pass Butterworth filter.

    Parameters:
    df (DataFrame): Input DataFrame with 'wavelength' and 'reflectance' columns.
    cutoff_frequency (float): Cutoff frequency for the high-pass filter (default=0.01).
    filter_order (int): Order of the Butterworth filter (default=3).

    Returns:
    DataFrame: DataFrame with an additional 'detrended' column.
    """
    # Ensure the DataFrame is sorted by wavelength
    df = df.sort_values(by="wavelength").reset_index(drop=True)
    
    # Create a copy to avoid modifying original
    df_result = df.copy()

    # Sampling frequency calculation from wavelength spacing
    sampling_interval = df['wavelength'].diff().mean()
    if sampling_interval <= 0:
        raise ValueError("Invalid wavelength spacing for detrending")
    
    sampling_frequency = 1 / sampling_interval

    # Normalize cutoff frequency
    nyquist_freq = 0.5 * sampling_frequency
    normal_cutoff = min(cutoff_frequency / nyquist_freq, 0.99)  # Ensure it's less than 1

    try:
        # High-pass Butterworth filter design
        b, a = butter(filter_order, normal_cutoff, btype='high', analog=False)

        # Apply filter using filtfilt for zero-phase filtering
        detrended_intensity = filtfilt(b, a, df['reflectance'])

        # Add detrended data to DataFrame
        df_result['detrended'] = detrended_intensity
    except Exception as e:
        st.warning(f"Detrending failed: {e}. Using original signal.")
        df_result['detrended'] = df_result['reflectance']

    return df_result


def detect_peaks(df, column='reflectance', prominence=0.005, height=None):
    """Detect peaks in the signal"""
    peaks_indices, properties = find_peaks(df[column], prominence=prominence, height=height)
    peaks_df = df.iloc[peaks_indices].copy()
    peaks_df['peak_prominence'] = properties.get('prominences', [0] * len(peaks_indices))
    return peaks_df.reset_index(drop=True)


def detect_valleys(df, column='reflectance', prominence=0.005):
    """Detect valleys in the signal"""
    valleys_indices, properties = find_peaks(-df[column], prominence=prominence)
    valleys_df = df.iloc[valleys_indices].copy()
    valleys_df['valley_prominence'] = properties.get('prominences', [0] * len(valleys_indices))
    return valleys_df.reset_index(drop=True)


def load_measurement_files(measurements_dir: pathlib.Path, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load all measurement files from the specified directory using enhanced loading."""
    measurements = {}
    
    if not measurements_dir.exists():
        st.warning(f"Measurements directory not found: {measurements_dir}")
        return measurements
    
    meas_config = config.get('measurements', {})
    file_pattern = meas_config.get('file_pattern', '*.csv')
    
    # Find all matching files
    pattern_path = measurements_dir / file_pattern
    file_paths = glob.glob(str(pattern_path))
    
    if not file_paths:
        st.warning(f"No measurement files found matching pattern: {file_pattern} in {measurements_dir}")
        return measurements
    
    for file_path in sorted(file_paths):
        try:
            file_name = pathlib.Path(file_path).stem
            
            # Use enhanced loading function
            meas_df = load_txt_file_enhanced(file_path)
            
            if len(meas_df) > 0:
                # Remove any NaN values
                meas_df = meas_df.dropna()
                measurements[file_name] = meas_df
                
        except Exception as e:
            st.warning(f"Error loading {file_path}: {e}")
    
    return measurements


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
    tab1, tab2, tab3 = st.tabs(["📊 Spectrum Comparison", "📈 Detrended Analysis", "⚙️ Parameters"])
    
    # Sidebar controls
    st.sidebar.markdown("## Layer Parameters")
    
    lipid_val = st.sidebar.slider(
        "Lipid thickness (nm)",
        min_value=float(lipid_cfg["min"]),
        max_value=float(lipid_cfg["max"]),
        value=float(default_lipid),
        step=float(lipid_cfg["step"]),
        format="%.0f",
    )

    aqueous_val = st.sidebar.slider(
        "Aqueous thickness (nm)",
        min_value=float(aqueous_cfg["min"]),
        max_value=float(aqueous_cfg["max"]),
        value=float(default_aqueous),
        step=float(aqueous_cfg["step"]),
        format="%.0f",
    )

    rough_val = st.sidebar.slider(
        "Mucus roughness (Å)",
        min_value=float(rough_cfg["min"]),
        max_value=float(rough_cfg["max"]),
        value=float(default_rough),
        step=float(rough_cfg["step"]),
        format="%.0f",
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
            theoretical_detrended = detrend_signal(theoretical_df, cutoff_freq)
            measured_detrended = detrend_signal(selected_measurement, cutoff_freq)
            
            # Detect peaks in detrended signals
            theo_peaks = detect_peaks(theoretical_detrended, 'detrended', peak_prominence)
            meas_peaks = detect_peaks(measured_detrended, 'detrended', peak_prominence)
            
            # Detect valleys in detrended signals
            theo_valleys = detect_valleys(theoretical_detrended, 'detrended', peak_prominence)
            meas_valleys = detect_valleys(measured_detrended, 'detrended', peak_prominence)
            
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
            theoretical_detrended = detrend_signal(theoretical_df, cutoff_freq)
            theo_peaks = detect_peaks(theoretical_detrended, 'detrended', peak_prominence)
            theo_valleys = detect_valleys(theoretical_detrended, 'detrended', peak_prominence)
            
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


if __name__ == "__main__":
    main()


