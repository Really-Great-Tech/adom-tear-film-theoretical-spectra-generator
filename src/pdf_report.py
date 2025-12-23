"""
PDF Report Generator for Tear Film Spectra Analysis.

Generates a multi-page PDF report showing the top N best fits from grid search,
with spectrum comparison and analysis plots side by side on each page.
"""

from __future__ import annotations

import io
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

# Page dimensions for landscape A4
PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
MARGIN = 0.5 * inch


def create_comparison_plot_for_pdf(
    wavelengths: np.ndarray,
    theoretical_spectrum: np.ndarray,
    measured_df: pd.DataFrame,
    lipid_nm: float,
    aqueous_nm: float,
    roughness_A: float,
    wl_min: float,
    wl_max: float,
    title: str = 'Spectrum Comparison'
) -> go.Figure:
    """Create spectrum comparison plot for PDF export."""
    fig = go.Figure()
    
    # Add measured spectrum
    fig.add_trace(go.Scatter(
        x=measured_df['wavelength'],
        y=measured_df['reflectance'],
        mode='lines',
        name='Measured',
        line=dict(color='red', width=2)
    ))
    
    # Add theoretical spectrum
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=theoretical_spectrum,
        mode='lines',
        name=f'Theoretical (L={lipid_nm:.0f}, A={aqueous_nm:.0f}, R={roughness_A:.0f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title='Wavelength (nm)',
        yaxis_title='Reflectance',
        xaxis_range=[wl_min, wl_max],
        template='plotly_white',
        width=500,
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01, font=dict(size=9))
    )
    
    return fig


def create_analysis_plot_for_pdf(
    measured_detrended: pd.DataFrame,
    theoretical_detrended: pd.DataFrame,
    meas_peaks: pd.DataFrame,
    theo_peaks: pd.DataFrame,
    meas_valleys: pd.DataFrame,
    theo_valleys: pd.DataFrame,
    wl_min: float,
    wl_max: float,
    title: str = 'Peak/Valley Analysis'
) -> go.Figure:
    """Create spectrum analysis plot showing detrended signals with peaks/valleys."""
    fig = go.Figure()
    
    # Measured detrended signal
    fig.add_trace(go.Scatter(
        x=measured_detrended['wavelength'],
        y=measured_detrended['detrended'],
        mode='lines',
        name='Measured',
        line=dict(color='blue', width=2)
    ))
    
    # Theoretical detrended signal
    fig.add_trace(go.Scatter(
        x=theoretical_detrended['wavelength'],
        y=theoretical_detrended['detrended'],
        mode='lines',
        name='Theoretical',
        line=dict(color='red', width=2, dash='dot')
    ))
    
    # Peaks and valleys
    if len(meas_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=meas_peaks['wavelength'], y=meas_peaks['detrended'],
            mode='markers', name='Meas Peaks',
            marker=dict(color='blue', size=6, symbol='circle')
        ))
    
    if len(theo_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=theo_peaks['wavelength'], y=theo_peaks['detrended'],
            mode='markers', name='Theo Peaks',
            marker=dict(color='red', size=6, symbol='circle')
        ))
    
    if len(meas_valleys) > 0:
        fig.add_trace(go.Scatter(
            x=meas_valleys['wavelength'], y=meas_valleys['detrended'],
            mode='markers', name='Meas Valleys',
            marker=dict(color='cyan', size=6, symbol='triangle-down')
        ))
    
    if len(theo_valleys) > 0:
        fig.add_trace(go.Scatter(
            x=theo_valleys['wavelength'], y=theo_valleys['detrended'],
            mode='markers', name='Theo Valleys',
            marker=dict(color='pink', size=6, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title='Wavelength (nm)',
        yaxis_title='Detrended Intensity',
        xaxis_range=[wl_min, wl_max],
        template='plotly_white',
        width=500,
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99, font=dict(size=8))
    )
    
    return fig


def fig_to_image_bytes(fig: go.Figure, scale: float = 2.0) -> bytes:
    """Convert Plotly figure to PNG bytes using kaleido."""
    return fig.to_image(format='png', scale=scale)


def generate_pdf_report(
    results_df: pd.DataFrame,
    measurement_file: str,
    measured_df: pd.DataFrame,
    single_spectrum_func,
    wavelengths: np.ndarray,
    analysis_cfg: Dict[str, Any],
    config: Dict[str, Any],
    detrend_func,
    detect_peaks_func,
    detect_valleys_func,
    prepare_theoretical_func,
    top_n: int = 10,
) -> bytes:
    """
    Generate a PDF report showing top N best fits.
    
    Each page contains:
    - Rank and parameters header
    - Spectrum comparison plot (left)
    - Peak/valley analysis plot (right)
    - Metrics table
    
    Returns PDF as bytes.
    """
    logger.info(f'📄 Generating PDF report for top {top_n} fits...')
    
    # Get wavelength range
    wavelength_range_cfg = analysis_cfg.get('wavelength_range', {})
    wl_min = float(wavelength_range_cfg.get('min', 600))
    wl_max = float(wavelength_range_cfg.get('max', 1120))
    
    # Get detrending params
    detrending_cfg = analysis_cfg.get('detrending', {})
    cutoff_freq = float(detrending_cfg.get('default_cutoff_frequency', 0.008))
    filter_order = int(detrending_cfg.get('filter_order', 3))
    
    # Get peak detection params
    peak_cfg = analysis_cfg.get('peak_detection', {})
    prominence = float(peak_cfg.get('default_prominence', 0.0001))
    
    # Filter measured data to wavelength range
    measured_filtered = measured_df[
        (measured_df['wavelength'] >= wl_min) & 
        (measured_df['wavelength'] <= wl_max)
    ].reset_index(drop=True)
    
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
    elements.append(Paragraph('Tear Film Spectra Analysis Report', title_style))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f'Measurement: {measurement_file}', subtitle_style))
    elements.append(Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', subtitle_style))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f'Top {min(top_n, len(results_df))} Best Fits', subtitle_style))
    elements.append(PageBreak())
    
    # Get top N results
    top_results = results_df.head(top_n)
    
    for rank, (idx, row) in enumerate(top_results.iterrows(), 1):
        logger.info(f'  📊 Processing rank {rank}/{len(top_results)}...')
        
        lipid_nm = float(row['lipid_nm'])
        aqueous_nm = float(row['aqueous_nm'])
        roughness_A = float(row['roughness_A'])
        
        # Generate theoretical spectrum
        theoretical_spectrum = single_spectrum_func(lipid_nm, aqueous_nm, roughness_A)
        
        # Filter theoretical to wavelength range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        theoretical_wl_filtered = wavelengths[mask]
        theoretical_spec_filtered = theoretical_spectrum[mask]
        
        # Create theoretical dataframe for detrending
        theoretical_df = pd.DataFrame({
            'wavelength': theoretical_wl_filtered,
            'reflectance': theoretical_spec_filtered
        })
        
        # Detrend signals
        measured_detrended = detrend_func(measured_filtered.copy(), cutoff_freq, filter_order)
        theoretical_detrended = detrend_func(theoretical_df.copy(), cutoff_freq, filter_order)
        
        # Detect peaks and valleys
        meas_peaks = detect_peaks_func(measured_detrended, 'detrended', prominence)
        theo_peaks = detect_peaks_func(theoretical_detrended, 'detrended', prominence)
        meas_valleys = detect_valleys_func(measured_detrended, 'detrended', prominence)
        theo_valleys = detect_valleys_func(theoretical_detrended, 'detrended', prominence)
        
        # Page header
        elements.append(Paragraph(f'Rank #{rank} - Best Fit Candidate', title_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Parameters
        param_text = (
            f'<b>Parameters:</b> Lipid = {lipid_nm:.1f} nm | '
            f'Aqueous = {aqueous_nm:.1f} nm | '
            f'Roughness = {roughness_A:.1f} Å'
        )
        elements.append(Paragraph(param_text, param_style))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Create plots
        comparison_fig = create_comparison_plot_for_pdf(
            theoretical_wl_filtered, theoretical_spec_filtered,
            measured_filtered, lipid_nm, aqueous_nm, roughness_A,
            wl_min, wl_max, title='Spectrum Comparison'
        )
        
        analysis_fig = create_analysis_plot_for_pdf(
            measured_detrended, theoretical_detrended,
            meas_peaks, theo_peaks, meas_valleys, theo_valleys,
            wl_min, wl_max, title='Peak/Valley Analysis'
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
        elements.append(Spacer(1, 0.2 * inch))
        
        # Metrics table
        score_columns = [col for col in row.index if col.startswith('score_')]
        metrics_data = [['Metric', 'Value']]
        
        for col in score_columns:
            metric_name = col.replace('score_', '').replace('_', ' ').title()
            value = row[col]
            metrics_data.append([metric_name, f'{value:.4f}'])
        
        # Add peak counts
        metrics_data.append(['Measured Peaks', str(len(meas_peaks))])
        metrics_data.append(['Theoretical Peaks', str(len(theo_peaks))])
        metrics_data.append(['Measured Valleys', str(len(meas_valleys))])
        metrics_data.append(['Theoretical Valleys', str(len(theo_valleys))])
        
        # Split metrics into two columns for compact display
        mid = (len(metrics_data) + 1) // 2
        left_metrics = metrics_data[:mid]
        right_metrics = metrics_data[mid:] if len(metrics_data) > mid else [['', '']]
        
        # Pad shorter list
        while len(right_metrics) < len(left_metrics):
            right_metrics.append(['', ''])
        while len(left_metrics) < len(right_metrics):
            left_metrics.append(['', ''])
        
        # Combine into 4-column table
        combined_metrics = []
        for left_row, right_row in zip(left_metrics, right_metrics):
            combined_metrics.append(left_row + right_row)
        
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
        
        # Page break (except for last page)
        if rank < len(top_results):
            elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    logger.info(f'✅ PDF report generated successfully ({len(top_results)} pages)')
    return pdf_buffer.getvalue()

