"""
PDF Report Generator for Tear Film Spectra Analysis.

Generates a multi-page PDF report showing the top N best fits,
with spectrum comparison and analysis plots side by side on each page.

This module is designed to be reusable across different apps (main streamlit, pyelli).
"""

from __future__ import annotations

import io
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

# Page dimensions for landscape A4
PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
MARGIN = 0.5 * inch


# =============================================================================
# Data Classes for Generic Report Interface
# =============================================================================

@dataclass
class FitResult:
    """Generic fit result that can be populated from different sources."""
    rank: int
    lipid_nm: float
    aqueous_nm: float
    roughness_or_mucus: float  # roughness_A for main app, mucus_nm for pyelli
    param_label: str  # 'Roughness (Å)' or 'Mucus (nm)'
    score: float
    metrics: Dict[str, float]  # Additional metrics to display
    # Plot data
    wavelengths: np.ndarray
    measured_spectrum: np.ndarray
    theoretical_spectrum: np.ndarray
    # Optional analysis data
    measured_detrended: Optional[pd.DataFrame] = None
    theoretical_detrended: Optional[pd.DataFrame] = None
    measured_peaks: Optional[pd.DataFrame] = None
    theoretical_peaks: Optional[pd.DataFrame] = None
    measured_valleys: Optional[pd.DataFrame] = None
    theoretical_valleys: Optional[pd.DataFrame] = None


# =============================================================================
# Plot Generation Functions
# =============================================================================

def create_comparison_plot_for_pdf(
    wavelengths: np.ndarray,
    theoretical_spectrum: np.ndarray,
    measured_spectrum: np.ndarray,
    lipid_nm: float,
    aqueous_nm: float,
    third_param: float,
    third_param_label: str,
    wl_min: float,
    wl_max: float,
    title: str = 'Spectrum Comparison'
) -> go.Figure:
    """Create spectrum comparison plot for PDF export."""
    fig = go.Figure()
    
    # Filter to wavelength range
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wl_filtered = wavelengths[mask]
    meas_filtered = measured_spectrum[mask] if len(measured_spectrum) == len(wavelengths) else measured_spectrum
    theo_filtered = theoretical_spectrum[mask] if len(theoretical_spectrum) == len(wavelengths) else theoretical_spectrum
    
    # Add measured spectrum
    fig.add_trace(go.Scatter(
        x=wl_filtered,
        y=meas_filtered[:len(wl_filtered)] if len(meas_filtered) > len(wl_filtered) else meas_filtered,
        mode='lines',
        name='Measured',
        line=dict(color='#2563eb', width=2)
    ))
    
    # Add theoretical spectrum
    third_val_str = f'{third_param:.0f}' if third_param >= 100 else f'{third_param:.1f}'
    fig.add_trace(go.Scatter(
        x=wl_filtered,
        y=theo_filtered[:len(wl_filtered)] if len(theo_filtered) > len(wl_filtered) else theo_filtered,
        mode='lines',
        name=f'Theoretical',
        line=dict(color='#059669', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_title='Wavelength (nm)',
        yaxis_title='Reflectance',
        xaxis_range=[wl_min, wl_max],
        template='plotly_white',
        width=480,
        height=320,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01, font=dict(size=8))
    )
    
    return fig


def create_analysis_plot_for_pdf(
    measured_detrended: Optional[pd.DataFrame],
    theoretical_detrended: Optional[pd.DataFrame],
    meas_peaks: Optional[pd.DataFrame],
    theo_peaks: Optional[pd.DataFrame],
    meas_valleys: Optional[pd.DataFrame],
    theo_valleys: Optional[pd.DataFrame],
    wl_min: float,
    wl_max: float,
    title: str = 'Peak/Valley Analysis'
) -> go.Figure:
    """Create spectrum analysis plot showing detrended signals with peaks/valleys."""
    fig = go.Figure()
    
    has_analysis_data = (measured_detrended is not None and theoretical_detrended is not None 
                         and len(measured_detrended) > 0 and len(theoretical_detrended) > 0)
    
    if has_analysis_data:
        # Measured detrended signal
        fig.add_trace(go.Scatter(
            x=measured_detrended['wavelength'],
            y=measured_detrended['detrended'],
            mode='lines',
            name='Measured',
            line=dict(color='#2563eb', width=2)
        ))
        
        # Theoretical detrended signal
        fig.add_trace(go.Scatter(
            x=theoretical_detrended['wavelength'],
            y=theoretical_detrended['detrended'],
            mode='lines',
            name='Theoretical',
            line=dict(color='#059669', width=2, dash='dot')
        ))
        
        # Peaks and valleys
        if meas_peaks is not None and len(meas_peaks) > 0:
            fig.add_trace(go.Scatter(
                x=meas_peaks['wavelength'], y=meas_peaks['detrended'],
                mode='markers', name='Meas Peaks',
                marker=dict(color='#2563eb', size=6, symbol='circle')
            ))
        
        if theo_peaks is not None and len(theo_peaks) > 0:
            fig.add_trace(go.Scatter(
                x=theo_peaks['wavelength'], y=theo_peaks['detrended'],
                mode='markers', name='Theo Peaks',
                marker=dict(color='#059669', size=6, symbol='circle')
            ))
        
        if meas_valleys is not None and len(meas_valleys) > 0:
            fig.add_trace(go.Scatter(
                x=meas_valleys['wavelength'], y=meas_valleys['detrended'],
                mode='markers', name='Meas Valleys',
                marker=dict(color='#60a5fa', size=6, symbol='triangle-down')
            ))
        
        if theo_valleys is not None and len(theo_valleys) > 0:
            fig.add_trace(go.Scatter(
                x=theo_valleys['wavelength'], y=theo_valleys['detrended'],
                mode='markers', name='Theo Valleys',
                marker=dict(color='#34d399', size=6, symbol='triangle-down')
            ))
        
        y_title = 'Detrended Intensity'
    else:
        # Fallback: show residual plot
        fig.add_annotation(
            text='Analysis data not available',
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#64748b')
        )
        y_title = 'N/A'
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_title='Wavelength (nm)',
        yaxis_title=y_title,
        xaxis_range=[wl_min, wl_max],
        template='plotly_white',
        width=480,
        height=320,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99, font=dict(size=7))
    )
    
    return fig


def create_residual_plot_for_pdf(
    wavelengths: np.ndarray,
    measured_spectrum: np.ndarray,
    theoretical_spectrum: np.ndarray,
    wl_min: float,
    wl_max: float,
    title: str = 'Residual Analysis'
) -> go.Figure:
    """Create residual plot for PDF export."""
    fig = go.Figure()
    
    # Filter to wavelength range
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wl_filtered = wavelengths[mask]
    meas_filtered = measured_spectrum[mask] if len(measured_spectrum) == len(wavelengths) else measured_spectrum
    theo_filtered = theoretical_spectrum[mask] if len(theoretical_spectrum) == len(wavelengths) else theoretical_spectrum
    
    # Ensure same length
    min_len = min(len(wl_filtered), len(meas_filtered), len(theo_filtered))
    residual = meas_filtered[:min_len] - theo_filtered[:min_len]
    
    fig.add_trace(go.Scatter(
        x=wl_filtered[:min_len],
        y=residual,
        mode='lines',
        name='Residual',
        line=dict(color='#d97706', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(251, 191, 36, 0.2)'
    ))
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_title='Wavelength (nm)',
        yaxis_title='Δ Reflectance',
        xaxis_range=[wl_min, wl_max],
        template='plotly_white',
        width=480,
        height=320,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False
    )
    
    return fig


def fig_to_image_bytes(fig: go.Figure, scale: float = 2.0) -> bytes:
    """Convert Plotly figure to PNG bytes using kaleido."""
    return fig.to_image(format='png', scale=scale)


# =============================================================================
# PDF Generation
# =============================================================================

def generate_pdf_report(
    fit_results: List[FitResult],
    measurement_file: str,
    wl_min: float = 600,
    wl_max: float = 1120,
    report_title: str = 'Tear Film Spectra Analysis Report',
    app_name: str = 'Tear Film Analysis',
) -> bytes:
    """
    Generate a PDF report showing fit results.
    
    Args:
        fit_results: List of FitResult objects with plot data
        measurement_file: Name of the measurement file
        wl_min, wl_max: Wavelength range for plots
        report_title: Title for the report
        app_name: Name of the app generating the report
    
    Returns:
        PDF as bytes
    """
    logger.info(f'📄 Generating PDF report for {len(fit_results)} fits...')
    
    # Create PDF buffer
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=8
    )
    param_style = ParagraphStyle(
        'ParamStyle',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_LEFT,
        spaceAfter=6
    )
    
    elements = []
    
    # Title page
    elements.append(Spacer(1, 2 * inch))
    elements.append(Paragraph(report_title, title_style))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f'Measurement: {measurement_file}', subtitle_style))
    elements.append(Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', subtitle_style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f'Top {len(fit_results)} Best Fits', subtitle_style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f'<i>Generated by {app_name}</i>', subtitle_style))
    elements.append(PageBreak())
    
    for fit in fit_results:
        logger.info(f'  📊 Processing rank {fit.rank}/{len(fit_results)}...')
        
        # Page header
        elements.append(Paragraph(f'Rank #{fit.rank} - Best Fit Candidate', title_style))
        elements.append(Spacer(1, 0.15 * inch))
        
        # Parameters
        third_val = f'{fit.roughness_or_mucus:.0f}' if fit.roughness_or_mucus >= 100 else f'{fit.roughness_or_mucus:.1f}'
        param_text = (
            f'<b>Parameters:</b> Lipid = {fit.lipid_nm:.1f} nm | '
            f'Aqueous = {fit.aqueous_nm:.1f} nm | '
            f'{fit.param_label} = {third_val}'
        )
        elements.append(Paragraph(param_text, param_style))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Create plots
        comparison_fig = create_comparison_plot_for_pdf(
            fit.wavelengths, fit.theoretical_spectrum, fit.measured_spectrum,
            fit.lipid_nm, fit.aqueous_nm, fit.roughness_or_mucus, fit.param_label,
            wl_min, wl_max, title='Spectrum Comparison'
        )
        
        # Use analysis plot if data available, otherwise residual
        if fit.measured_detrended is not None and fit.theoretical_detrended is not None:
            analysis_fig = create_analysis_plot_for_pdf(
                fit.measured_detrended, fit.theoretical_detrended,
                fit.measured_peaks, fit.theoretical_peaks,
                fit.measured_valleys, fit.theoretical_valleys,
                wl_min, wl_max, title='Peak/Valley Analysis'
            )
        else:
            analysis_fig = create_residual_plot_for_pdf(
                fit.wavelengths, fit.measured_spectrum, fit.theoretical_spectrum,
                wl_min, wl_max, title='Residual Analysis'
            )
        
        # Convert to images
        comparison_img_bytes = fig_to_image_bytes(comparison_fig)
        analysis_img_bytes = fig_to_image_bytes(analysis_fig)
        
        comparison_img = Image(io.BytesIO(comparison_img_bytes), width=4.5*inch, height=3*inch)
        analysis_img = Image(io.BytesIO(analysis_img_bytes), width=4.5*inch, height=3*inch)
        
        # Create side-by-side table for plots
        plot_table = Table(
            [[comparison_img, analysis_img]],
            colWidths=[4.7*inch, 4.7*inch]
        )
        plot_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(plot_table)
        elements.append(Spacer(1, 0.15 * inch))
        
        # Metrics table
        metrics_data = [['Metric', 'Value']]
        metrics_data.append(['Composite Score', f'{fit.score:.4f}'])
        
        for metric_name, metric_val in fit.metrics.items():
            display_name = metric_name.replace('_', ' ').title()
            if isinstance(metric_val, float):
                metrics_data.append([display_name, f'{metric_val:.4f}'])
            else:
                metrics_data.append([display_name, str(metric_val)])
        
        # Split into two columns
        mid = (len(metrics_data) + 1) // 2
        left_metrics = metrics_data[:mid]
        right_metrics = metrics_data[mid:] if len(metrics_data) > mid else [['', '']]
        
        while len(right_metrics) < len(left_metrics):
            right_metrics.append(['', ''])
        while len(left_metrics) < len(right_metrics):
            left_metrics.append(['', ''])
        
        combined_metrics = [left + right for left, right in zip(left_metrics, right_metrics)]
        
        metrics_table = Table(combined_metrics, colWidths=[1.8*inch, 1.2*inch, 1.8*inch, 1.2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('BACKGROUND', (2, 0), (3, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(metrics_table)
        
        # Page break except for last
        if fit.rank < len(fit_results):
            elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    logger.info(f'✅ PDF report generated successfully ({len(fit_results)} pages)')
    return pdf_buffer.getvalue()


# =============================================================================
# Convenience Functions for Different Apps
# =============================================================================

def generate_main_app_pdf_report(
    results_df: pd.DataFrame,
    measurement_file: str,
    measured_df: pd.DataFrame,
    single_spectrum_func: Callable,
    wavelengths: np.ndarray,
    analysis_cfg: Dict[str, Any],
    detrend_func: Callable,
    detect_peaks_func: Callable,
    detect_valleys_func: Callable,
    top_n: int = 10,
) -> bytes:
    """
    Generate PDF report for the main streamlit app.
    
    This wraps the generic generate_pdf_report with main app specifics.
    """
    wavelength_range_cfg = analysis_cfg.get('wavelength_range', {})
    wl_min = float(wavelength_range_cfg.get('min', 600))
    wl_max = float(wavelength_range_cfg.get('max', 1120))
    
    detrending_cfg = analysis_cfg.get('detrending', {})
    cutoff_freq = float(detrending_cfg.get('default_cutoff_frequency', 0.008))
    filter_order = int(detrending_cfg.get('filter_order', 3))
    
    peak_cfg = analysis_cfg.get('peak_detection', {})
    prominence = float(peak_cfg.get('default_prominence', 0.0001))
    
    # Filter measured data
    measured_filtered = measured_df[
        (measured_df['wavelength'] >= wl_min) & 
        (measured_df['wavelength'] <= wl_max)
    ].reset_index(drop=True)
    
    fit_results = []
    top_results = results_df.head(top_n)
    
    for rank, (idx, row) in enumerate(top_results.iterrows(), 1):
        lipid_nm = float(row['lipid_nm'])
        aqueous_nm = float(row['aqueous_nm'])
        roughness_A = float(row['roughness_A'])
        
        # Generate theoretical spectrum
        theoretical_spectrum = single_spectrum_func(lipid_nm, aqueous_nm, roughness_A)
        
        # Filter to wavelength range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        wl_filtered = wavelengths[mask]
        theo_filtered = theoretical_spectrum[mask]
        
        # Create dataframe for detrending
        theoretical_df = pd.DataFrame({
            'wavelength': wl_filtered,
            'reflectance': theo_filtered
        })
        
        # Detrend
        measured_detrended = detrend_func(measured_filtered.copy(), cutoff_freq, filter_order)
        theoretical_detrended = detrend_func(theoretical_df.copy(), cutoff_freq, filter_order)
        
        # Detect peaks/valleys
        meas_peaks = detect_peaks_func(measured_detrended, 'detrended', prominence)
        theo_peaks = detect_peaks_func(theoretical_detrended, 'detrended', prominence)
        meas_valleys = detect_valleys_func(measured_detrended, 'detrended', prominence)
        theo_valleys = detect_valleys_func(theoretical_detrended, 'detrended', prominence)
        
        # Collect metrics
        metrics = {}
        score_cols = [c for c in row.index if c.startswith('score_') and c != 'score_composite']
        for col in score_cols:
            metrics[col.replace('score_', '')] = float(row[col])
        metrics['measured_peaks'] = len(meas_peaks)
        metrics['theoretical_peaks'] = len(theo_peaks)
        metrics['measured_valleys'] = len(meas_valleys)
        metrics['theoretical_valleys'] = len(theo_valleys)
        
        fit_results.append(FitResult(
            rank=rank,
            lipid_nm=lipid_nm,
            aqueous_nm=aqueous_nm,
            roughness_or_mucus=roughness_A,
            param_label='Roughness (Å)',
            score=float(row.get('score_composite', 0)),
            metrics=metrics,
            wavelengths=wl_filtered,
            measured_spectrum=measured_filtered['reflectance'].values,
            theoretical_spectrum=theo_filtered,
            measured_detrended=measured_detrended,
            theoretical_detrended=theoretical_detrended,
            measured_peaks=meas_peaks,
            theoretical_peaks=theo_peaks,
            measured_valleys=meas_valleys,
            theoretical_valleys=theo_valleys,
        ))
    
    return generate_pdf_report(
        fit_results,
        measurement_file,
        wl_min=wl_min,
        wl_max=wl_max,
        report_title='Tear Film Spectra Analysis Report',
        app_name='Tear Film Spectra Explorer'
    )


def generate_pyelli_pdf_report(
    autofit_results: List,  # List of PyElliResult objects
    measurement_file: str,
    measured_wavelengths: np.ndarray,
    measured_spectrum: np.ndarray,
    wl_min: float = 600,
    wl_max: float = 1200,
    top_n: int = 10,
) -> bytes:
    """
    Generate PDF report for the PyElli exploration app.
    
    Args:
        autofit_results: List of PyElliResult objects from grid search
        measurement_file: Name of the measurement file
        measured_wavelengths: Wavelength array of measured data
        measured_spectrum: Measured reflectance values
        wl_min, wl_max: Wavelength range for plots
        top_n: Number of top results to include
    """
    fit_results = []
    
    for rank, result in enumerate(autofit_results[:top_n], 1):
        # Filter to wavelength range
        mask = (result.wavelengths >= wl_min) & (result.wavelengths <= wl_max)
        wl_filtered = result.wavelengths[mask]
        theo_filtered = result.theoretical_spectrum[mask]
        
        # Also filter measured
        meas_mask = (measured_wavelengths >= wl_min) & (measured_wavelengths <= wl_max)
        meas_wl_filtered = measured_wavelengths[meas_mask]
        meas_filtered = measured_spectrum[meas_mask]
        
        # Interpolate measured to match theoretical wavelengths
        meas_interp = np.interp(wl_filtered, meas_wl_filtered, meas_filtered)
        
        metrics = {
            'correlation': result.correlation,
            'rmse': result.rmse,
            'crossings': result.crossing_count,
        }
        
        fit_results.append(FitResult(
            rank=rank,
            lipid_nm=result.lipid_nm,
            aqueous_nm=result.aqueous_nm,
            roughness_or_mucus=result.mucus_nm,
            param_label='Mucus (nm)',
            score=result.score,
            metrics=metrics,
            wavelengths=wl_filtered,
            measured_spectrum=meas_interp,
            theoretical_spectrum=theo_filtered,
            # PyElli doesn't have detrended analysis
            measured_detrended=None,
            theoretical_detrended=None,
            measured_peaks=None,
            theoretical_peaks=None,
            measured_valleys=None,
            theoretical_valleys=None,
        ))
    
    return generate_pdf_report(
        fit_results,
        measurement_file,
        wl_min=wl_min,
        wl_max=wl_max,
        report_title='PyElli Auto-Fit Analysis Report',
        app_name='PyElli Exploration Tool'
    )
