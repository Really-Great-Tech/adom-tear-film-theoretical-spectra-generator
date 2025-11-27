"""
PyElli Exploration App

A Streamlit application demonstrating pyElli's capabilities for tear film
interferometry analysis using ADOM sample data.

This is an exploration tool for evaluating pyElli's potential integration
into the AdOM-TFI workflow.

Run with: streamlit run app.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add parent paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pyelli_exploration.pyelli_utils import (
    load_measured_spectrum,
    load_bestfit_spectrum,
    load_material_data,
    get_sample_data_paths,
    get_available_materials,
    calculate_residual,
    calculate_correlation,
    interpolate_to_common_wavelengths,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title='PyElli Exploration | AdOM-TFI',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Clean Light Theme CSS
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main app styling */
    .stApp {
        background: #ffffff;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Typography */
    html, body, [class*="st-"] {
        font-family: 'DM Sans', sans-serif;
        color: #1e293b;
    }
    
    h1, h2, h3, h4 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        color: #0f172a;
    }
    
    h1 { font-size: 2.25rem !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.5rem !important; color: #1e40af; }
    h3 { font-size: 1.15rem !important; color: #334155; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1e40af !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #475569 !important;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 18px;
        color: #64748b;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #fff;
        color: #1e40af;
    }
    
    .stTabs [aria-selected="true"] {
        background: #fff !important;
        color: #1e40af !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem !important;
        font-weight: 600;
        color: #1e40af;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%) !important;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #1e40af;
        font-weight: 600;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #f8fafc;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .stRadio label {
        color: #334155 !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #fff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        color: #334155 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1e40af;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(30, 64, 175, 0.2);
    }
    
    .stButton > button:hover {
        background: #1e3a8a;
        box-shadow: 0 4px 8px rgba(30, 64, 175, 0.3);
    }
    
    /* Success alerts */
    .stAlert {
        border-radius: 8px !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Separators */
    hr {
        border-color: #e2e8f0 !important;
    }
    
    /* Custom classes */
    .hero-text {
        font-size: 1.05rem;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .gradient-text {
        color: #1e40af;
        font-weight: 700;
    }
    
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }
    
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
</style>
''', unsafe_allow_html=True)


# =============================================================================
# Path Configuration
# =============================================================================

SAMPLE_DATA_PATH = PROJECT_ROOT / 'exploration' / 'sample_data'
MATERIALS_PATH = PROJECT_ROOT / 'data' / 'Materials'


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown('''
    <div style="text-align: center; padding: 16px 0;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">🔬</div>
        <h2 style="margin: 0; font-size: 1.3rem; color: #1e40af;">PyElli Explorer</h2>
        <p style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Thin Film Optics Analysis</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<hr style="margin: 12px 0; border-color: #e2e8f0;">', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="card" style="padding: 14px;">
        <p style="color: #1e40af; font-weight: 600; margin-bottom: 10px; font-size: 0.8rem;">✨ KEY CAPABILITIES</p>
        <div style="display: flex; flex-direction: column; gap: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">📐</span>
                <span style="color: #475569; font-size: 0.82rem;">Transfer Matrix Method</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">🌈</span>
                <span style="color: #475569; font-size: 0.82rem;">Dispersion Modeling</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">📊</span>
                <span style="color: #475569; font-size: 0.82rem;">Multi-Layer Optics</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">⚡</span>
                <span style="color: #475569; font-size: 0.82rem;">Spectral Fitting</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<hr style="margin: 12px 0; border-color: #e2e8f0;">', unsafe_allow_html=True)
    
    # Check data availability
    samples = get_sample_data_paths(SAMPLE_DATA_PATH)
    materials = get_available_materials(MATERIALS_PATH)
    
    st.markdown('''
    <p style="color: #1e40af; font-weight: 600; margin-bottom: 10px; font-size: 0.8rem;">📂 DATA STATUS</p>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #16a34a;">{len(samples["good_fit"])}</div>
            <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">Good Fits</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #dc2626;">{len(samples["bad_fit"])}</div>
            <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">Bad Fits</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px; text-align: center; margin-top: 8px;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #1e40af;">{len(materials)}</div>
        <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">Materials</div>
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# Main Content
# =============================================================================

st.markdown('''
<div style="text-align: center; padding: 16px 0 32px 0;">
    <h1 style="font-size: 2.2rem; margin-bottom: 12px; color: #0f172a;">
        🔬 PyElli Exploration
    </h1>
    <p class="hero-text" style="max-width: 650px; margin: 0 auto; color: #64748b;">
        Evaluate <span style="color: #1e40af; font-weight: 600;">pyElli</span> for tear film interferometry modeling. 
        Explore transfer matrix calculations, material dispersion, and spectral fitting.
    </p>
</div>
''', unsafe_allow_html=True)

# Custom Plotly theme - Clean white
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': '#ffffff',
        'plot_bgcolor': '#ffffff',
        'font': {'family': 'DM Sans, sans-serif', 'color': '#334155'},
        'title': {'font': {'family': 'DM Sans, sans-serif', 'color': '#1e40af'}},
        'xaxis': {
            'gridcolor': '#e5e7eb',
            'linecolor': '#d1d5db',
            'tickfont': {'color': '#6b7280'},
            'title': {'font': {'color': '#374151'}}
        },
        'yaxis': {
            'gridcolor': '#e5e7eb',
            'linecolor': '#d1d5db',
            'tickfont': {'color': '#6b7280'},
            'title': {'font': {'color': '#374151'}}
        },
        'legend': {
            'bgcolor': '#ffffff',
            'bordercolor': '#e5e7eb',
            'font': {'color': '#374151'}
        }
    }
}

# Color palette - Light theme friendly
COLORS = {
    'primary': '#1e40af',      # Deep blue
    'secondary': '#7c3aed',    # Purple
    'accent': '#0891b2',       # Cyan
    'success': '#059669',      # Green
    'warning': '#d97706',      # Amber
    'measured': '#2563eb',     # Blue for measured data
    'theoretical': '#7c3aed',  # Purple for theoretical
    'bestfit': '#db2777',      # Pink for bestfit
    'residual': '#d97706',     # Amber for residual
}

tabs = st.tabs([
    '📊 Sample Data Viewer',
    '🌈 Material Properties',
    '🔧 PyElli Structure Demo',
    '📈 Fitting Comparison',
    '📚 Integration Guide'
])


# =============================================================================
# Tab 1: Sample Data Viewer
# =============================================================================

with tabs[0]:
    st.markdown('''
    <div style="margin-bottom: 24px;">
        <h2>📊 Sample Data Viewer</h2>
        <p style="color: #94a3b8;">Explore measured spectra and their corresponding BestFit theoretical matches from ADOM's LTA software.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('### Sample Selection')
        
        fit_category = st.radio(
            'Fit Quality',
            ['good_fit', 'bad_fit'],
            format_func=lambda x: '✅ Good Fits' if x == 'good_fit' else '❌ Bad Fits'
        )
        
        available_samples = list(samples[fit_category].keys())
        selected_sample = st.selectbox(
            'Sample ID',
            available_samples,
            format_func=lambda x: f'Sample {x}'
        )
        
        show_residual = st.checkbox('Show Residual Plot', value=True)
        wavelength_range = st.slider(
            'Wavelength Range (nm)',
            400, 1200,
            (600, 1100),
            step=10
        )
    
    with col2:
        if selected_sample:
            sample_info = samples[fit_category][selected_sample]
            
            # Load spectra
            if sample_info['measured'] and sample_info['bestfit']:
                measured_wl, measured_refl = load_measured_spectrum(sample_info['measured'])
                bestfit_wl, bestfit_refl = load_bestfit_spectrum(sample_info['bestfit'])
                
                # Interpolate to common wavelengths
                common_wl, meas_interp, best_interp = interpolate_to_common_wavelengths(
                    measured_wl, measured_refl,
                    bestfit_wl, bestfit_refl
                )
                
                # Filter to wavelength range
                mask = (common_wl >= wavelength_range[0]) & (common_wl <= wavelength_range[1])
                common_wl_filtered = common_wl[mask]
                meas_filtered = meas_interp[mask]
                best_filtered = best_interp[mask]
                
                # Calculate metrics
                residual = calculate_residual(meas_filtered, best_filtered)
                correlation = calculate_correlation(meas_filtered, best_filtered)
                
                # Create figure
                if show_residual:
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=['', 'Residual']
                    )
                    
                    # Main spectra plot
                    fig.add_trace(
                        go.Scatter(
                            x=common_wl_filtered, y=meas_filtered,
                            mode='lines', name='Measured',
                            line=dict(color=COLORS['measured'], width=2.5)
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=common_wl_filtered, y=best_filtered,
                            mode='lines', name='BestFit (LTA)',
                            line=dict(color=COLORS['bestfit'], width=2.5, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # Residual plot
                    residuals = meas_filtered - best_filtered
                    fig.add_trace(
                        go.Scatter(
                            x=common_wl_filtered, y=residuals,
                            mode='lines', name='Residual',
                            line=dict(color=COLORS['residual'], width=1.5),
                            fill='tozeroy',
                            fillcolor='rgba(251, 191, 36, 0.15)'
                        ),
                        row=2, col=1
                    )
                    fig.add_hline(y=0, line_dash='dot', line_color='rgba(148, 163, 184, 0.3)', row=2, col=1)
                    
                    fig.update_xaxes(title_text='Wavelength (nm)', row=2, col=1, **PLOTLY_TEMPLATE['layout']['xaxis'])
                    fig.update_yaxes(title_text='Reflectance', row=1, col=1, **PLOTLY_TEMPLATE['layout']['yaxis'])
                    fig.update_yaxes(title_text='Δ', row=2, col=1, **PLOTLY_TEMPLATE['layout']['yaxis'])
                    
                else:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=common_wl_filtered, y=meas_filtered,
                            mode='lines', name='Measured',
                            line=dict(color=COLORS['measured'], width=2.5)
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=common_wl_filtered, y=best_filtered,
                            mode='lines', name='BestFit (LTA)',
                            line=dict(color=COLORS['bestfit'], width=2.5, dash='dash')
                        )
                    )
                    fig.update_xaxes(title_text='Wavelength (nm)', **PLOTLY_TEMPLATE['layout']['xaxis'])
                    fig.update_yaxes(title_text='Reflectance', **PLOTLY_TEMPLATE['layout']['yaxis'])
                
                fig.update_layout(
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(family='DM Sans, sans-serif', color='#374151'),
                    height=550 if show_residual else 400,
                    margin=dict(t=30, b=30, l=60, r=30),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1,
                        bgcolor='#ffffff',
                        bordercolor='#e5e7eb',
                        font=dict(color='#374151')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics display with custom styling
                st.markdown('''
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 20px;">
                ''', unsafe_allow_html=True)
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.markdown(f'''
                    <div class="stat-card">
                        <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">RMS Residual</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: #2563eb;">{residual:.6f}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with metric_cols[1]:
                    st.markdown(f'''
                    <div class="stat-card">
                        <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Correlation</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: #7c3aed;">{correlation:.4f}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with metric_cols[2]:
                    st.markdown(f'''
                    <div class="stat-card">
                        <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Data Points</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: #0891b2;">{len(common_wl_filtered)}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with metric_cols[3]:
                    quality_color = '#059669' if fit_category == 'good_fit' else '#dc2626'
                    quality_text = 'Good' if fit_category == 'good_fit' else 'Poor'
                    quality_icon = '●' if fit_category == 'good_fit' else '●'
                    st.markdown(f'''
                    <div class="stat-card">
                        <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Fit Quality</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: {quality_color};">{quality_icon} {quality_text}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning('⚠️ Missing spectrum files for this sample')


# =============================================================================
# Tab 2: Material Properties
# =============================================================================

with tabs[1]:
    st.markdown('''
    <div style="margin-bottom: 24px;">
        <h2>🌈 Material Optical Properties</h2>
        <p style="color: #94a3b8;">Visualize refractive index (n) and extinction coefficient (k) dispersion data for tear film materials.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Material selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('### Material Selection')
        
        # Categorize materials
        tear_film_materials = {
            'lipid_05-02621extrapolated': 'Lipid Layer',
            'water_Bashkatov1353extrapolated': 'Aqueous (Water)',
            'struma_Bashkatov140extrapolated': 'Mucus/Stroma',
        }
        
        substrate_materials = {
            'BK7': 'BK7 Glass',
            'SiO2': 'Silicon Dioxide',
        }
        
        material_category = st.radio(
            'Category',
            ['Tear Film', 'Substrates', 'All Materials']
        )
        
        if material_category == 'Tear Film':
            available_mats = [m for m in tear_film_materials.keys() if m in materials]
            selected_materials = st.multiselect(
                'Select Materials',
                available_mats,
                default=available_mats[:3],
                format_func=lambda x: tear_film_materials.get(x, x)
            )
        elif material_category == 'Substrates':
            available_mats = [m for m in substrate_materials.keys() if m in materials]
            selected_materials = st.multiselect(
                'Select Materials',
                available_mats,
                default=available_mats[:2] if available_mats else [],
                format_func=lambda x: substrate_materials.get(x, x)
            )
        else:
            selected_materials = st.multiselect(
                'Select Materials',
                list(materials.keys()),
                default=list(tear_film_materials.keys())[:3]
            )
        
        show_extinction = st.checkbox('Show Extinction (k)', value=False)
        
        wl_range = st.slider(
            'Wavelength Range (nm)',
            200, 1200,
            (400, 1100),
            step=10
        )
    
    with col2:
        if selected_materials:
            fig = make_subplots(
                rows=2 if show_extinction else 1, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=['Refractive Index (n)', 'Extinction Coefficient (k)'] if show_extinction else ['Refractive Index (n)']
            )
            
            colors = ['#2563eb', '#db2777', '#059669', '#d97706', '#7c3aed', '#dc2626']  # Light-theme friendly palette
            
            for i, mat_name in enumerate(selected_materials):
                mat_df = load_material_data(materials[mat_name])
                
                # Filter wavelength range
                mask = (mat_df['wavelength_nm'] >= wl_range[0]) & (mat_df['wavelength_nm'] <= wl_range[1])
                mat_df = mat_df[mask]
                
                display_name = tear_film_materials.get(mat_name, mat_name)
                
                # Add n trace
                fig.add_trace(
                    go.Scatter(
                        x=mat_df['wavelength_nm'], y=mat_df['n'],
                        mode='lines', name=f'{display_name}',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ),
                    row=1, col=1
                )
                
                # Add k trace if enabled
                if show_extinction:
                    fig.add_trace(
                        go.Scatter(
                            x=mat_df['wavelength_nm'], y=mat_df['k'],
                            mode='lines', name=f'{display_name} (k)',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
            
            fig.update_xaxes(title_text='Wavelength (nm)', row=2 if show_extinction else 1, col=1, **PLOTLY_TEMPLATE['layout']['xaxis'])
            fig.update_yaxes(title_text='Refractive Index (n)', row=1, col=1, **PLOTLY_TEMPLATE['layout']['yaxis'])
            if show_extinction:
                fig.update_yaxes(title_text='Extinction (k)', row=2, col=1, **PLOTLY_TEMPLATE['layout']['yaxis'])
            
            fig.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#ffffff',
                font=dict(family='DM Sans, sans-serif', color='#374151'),
                height=550 if show_extinction else 400,
                margin=dict(t=40, b=40, l=60, r=30),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor='#ffffff',
                    bordercolor='#e5e7eb',
                    font=dict(color='#374151')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Material info table
            st.markdown('### Material Summary')
            summary_data = []
            for mat_name in selected_materials:
                mat_df = load_material_data(materials[mat_name])
                summary_data.append({
                    'Material': tear_film_materials.get(mat_name, mat_name),
                    'λ Min (nm)': f'{mat_df["wavelength_nm"].min():.1f}',
                    'λ Max (nm)': f'{mat_df["wavelength_nm"].max():.1f}',
                    'n @ 550nm': f'{np.interp(550, mat_df["wavelength_nm"], mat_df["n"]):.4f}',
                    'n @ 800nm': f'{np.interp(800, mat_df["wavelength_nm"], mat_df["n"]):.4f}',
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        else:
            st.info('👈 Select materials from the sidebar to visualize their optical properties')


# =============================================================================
# Tab 3: PyElli Structure Demo
# =============================================================================

with tabs[2]:
    st.markdown('''
    <div style="margin-bottom: 24px;">
        <h2>🔧 PyElli Structure Builder</h2>
        <p style="color: #94a3b8;">Build multi-layer thin film structures and calculate optical responses using the Transfer Matrix Method.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Check if pyElli is available
    try:
        import elli
        pyelli_available = True
    except ImportError:
        pyelli_available = False
    
    if not pyelli_available:
        st.warning('''
        ⚠️ **pyElli not installed**
        
        To run the full pyElli demos, install it with:
        ```bash
        pip install pyElli
        ```
        
        The demo below shows the conceptual approach using a simplified model.
        ''')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('### Layer Configuration')
        
        st.markdown('#### Tear Film Structure')
        st.markdown('`Air → Lipid → Aqueous → Mucus → Eye`')
        
        lipid_thickness = st.slider(
            'Lipid Thickness (nm)',
            10, 200, 80,
            help='Typical range: 40-100 nm',
            key='demo_lipid'
        )
        
        aqueous_thickness = st.slider(
            'Aqueous Thickness (nm)',
            500, 5000, 2337,
            step=50,
            help='Typical range: 1000-5000 nm',
            key='demo_aqueous'
        )
        
        mucus_thickness = st.slider(
            'Mucus Thickness (nm)',
            100, 1000, 500,
            step=50,
            help='Typical range: 200-800 nm',
            key='demo_mucus'
        )
        
        st.markdown('#### Calculation Parameters')
        
        calc_wavelength_range = st.slider(
            'Wavelength Range (nm)',
            400, 1200,
            (600, 1100),
            step=10,
            key='demo_wavelength'
        )
        
        num_points = st.slider(
            'Number of Points',
            100, 1000, 500,
            step=50,
            key='demo_num_points'
        )
    
    with col2:
        st.markdown('### Theoretical Spectrum')
        
        # Load material data
        lipid_df = load_material_data(materials['lipid_05-02621extrapolated'])
        water_df = load_material_data(materials['water_Bashkatov1353extrapolated'])
        mucus_df = load_material_data(materials['struma_Bashkatov140extrapolated'])
        
        # Create wavelength array
        wavelengths = np.linspace(
            calc_wavelength_range[0],
            calc_wavelength_range[1],
            num_points
        )
        
        # Interpolate material properties
        def get_nk(mat_df, wavelengths):
            n = np.interp(wavelengths, mat_df['wavelength_nm'], mat_df['n'])
            k = np.interp(wavelengths, mat_df['wavelength_nm'], mat_df['k'])
            return n, k
        
        lipid_n, lipid_k = get_nk(lipid_df, wavelengths)
        water_n, water_k = get_nk(water_df, wavelengths)
        mucus_n, mucus_k = get_nk(mucus_df, wavelengths)
        
        # Simple multi-layer reflectance calculation
        # Using transfer matrix method approximation
        def transfer_matrix_reflectance(wavelengths, layers):
            """
            Calculate reflectance using transfer matrix method.
            Each layer is (n_array, k_array, thickness_nm)
            """
            n_air = 1.0
            reflectance = np.zeros_like(wavelengths)
            
            for i, wl in enumerate(wavelengths):
                # Build complex refractive indices
                N = [n_air]  # Start with air
                d = [0]  # Air has no thickness
                
                for n_arr, k_arr, thickness in layers:
                    N.append(n_arr[i] + 1j * k_arr[i])
                    d.append(thickness)
                
                # Substrate (approximate as semi-infinite mucus)
                N.append(N[-1])
                
                # Transfer matrix calculation
                M = np.eye(2, dtype=complex)
                
                for j in range(1, len(N) - 1):
                    # Interface matrix
                    r_jk = (N[j-1] - N[j]) / (N[j-1] + N[j])
                    t_jk = 2 * N[j-1] / (N[j-1] + N[j])
                    
                    I_jk = np.array([
                        [1, r_jk],
                        [r_jk, 1]
                    ], dtype=complex) / t_jk
                    
                    # Propagation matrix
                    delta = 2 * np.pi * N[j] * d[j] / wl
                    L_j = np.array([
                        [np.exp(-1j * delta), 0],
                        [0, np.exp(1j * delta)]
                    ], dtype=complex)
                    
                    M = M @ I_jk @ L_j
                
                # Final interface
                r_final = (N[-2] - N[-1]) / (N[-2] + N[-1])
                t_final = 2 * N[-2] / (N[-2] + N[-1])
                I_final = np.array([
                    [1, r_final],
                    [r_final, 1]
                ], dtype=complex) / t_final
                
                M = M @ I_final
                
                # Reflectance from M matrix
                r = M[1, 0] / M[0, 0]
                reflectance[i] = np.abs(r) ** 2
            
            return reflectance
        
        # Calculate reflectance
        layers = [
            (lipid_n, lipid_k, lipid_thickness),
            (water_n, water_k, aqueous_thickness),
            (mucus_n, mucus_k, mucus_thickness),
        ]
        
        theoretical_reflectance = transfer_matrix_reflectance(wavelengths, layers)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=wavelengths, y=theoretical_reflectance,
                mode='lines',
                name='Theoretical (TMM)',
                line=dict(color=COLORS['theoretical'], width=2.5),
                fill='tozeroy',
                fillcolor='rgba(124, 58, 237, 0.1)'
            )
        )
        
        fig.update_layout(
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family='DM Sans, sans-serif', color='#374151'),
            height=400,
            margin=dict(t=60, b=40, l=60, r=30),
            title=dict(
                text=f'<b>Theoretical Reflectance</b><br><span style="font-size:12px;color:#64748b">Lipid: {lipid_thickness}nm | Aqueous: {aqueous_thickness}nm | Mucus: {mucus_thickness}nm</span>',
                font=dict(size=16, color='#1e40af')
            ),
            xaxis=dict(
                gridcolor='#e5e7eb',
                linecolor='#d1d5db',
                tickfont=dict(color='#6b7280'),
                title=dict(text='Wavelength (nm)', font=dict(color='#374151'))
            ),
            yaxis=dict(
                gridcolor='#e5e7eb',
                linecolor='#d1d5db',
                tickfont=dict(color='#6b7280'),
                title=dict(text='Reflectance', font=dict(color='#374151'))
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Structure visualization
        st.markdown('### Layer Structure Diagram')
        
        # Create a simple bar chart to visualize layer thicknesses
        fig_structure = go.Figure()
        
        layer_names = ['Air', 'Lipid', 'Aqueous', 'Mucus', 'Eye']
        layer_thicknesses = [0, lipid_thickness, aqueous_thickness, mucus_thickness, 0]
        layer_colors = ['#475569', '#fbbf24', '#60a5fa', '#34d399', '#f472b6']
        
        # Vertical bar representation
        cumulative = 0
        for name, thickness, color in zip(layer_names, layer_thicknesses, layer_colors):
            if thickness > 0:
                fig_structure.add_trace(
                    go.Bar(
                        x=[name],
                        y=[thickness],
                        name=name,
                        marker_color=color,
                        text=f'{thickness} nm',
                        textposition='inside'
                    )
                )
        
        fig_structure.update_layout(
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            font=dict(family='DM Sans, sans-serif', color='#374151'),
            height=300,
            margin=dict(t=50, b=40, l=60, r=30),
            showlegend=False,
            title=dict(text='Layer Thickness Comparison', font=dict(size=14, color='#1e40af')),
            xaxis=dict(
                gridcolor='rgba(139, 92, 246, 0.1)',
                linecolor='rgba(139, 92, 246, 0.3)',
                tickfont=dict(color='#94a3b8')
            ),
            yaxis=dict(
                gridcolor='#e5e7eb',
                linecolor='#d1d5db',
                tickfont=dict(color='#6b7280'),
                title=dict(text='Thickness (nm)', font=dict(color='#374151'))
            )
        )
        
        st.plotly_chart(fig_structure, use_container_width=True)


# =============================================================================
# Tab 4: Fitting Comparison
# =============================================================================

with tabs[3]:
    st.markdown('''
    <div style="margin-bottom: 24px;">
        <h2>📈 Fitting Comparison</h2>
        <p style="color: #94a3b8;">Compare pyElli-generated theoretical spectra with measured data and LTA BestFit results.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('### Sample Selection')
        
        compare_category = st.radio(
            'Fit Category',
            ['good_fit', 'bad_fit'],
            format_func=lambda x: '✅ Good Fits' if x == 'good_fit' else '❌ Bad Fits',
            key='compare_category'
        )
        
        compare_sample = st.selectbox(
            'Sample',
            list(samples[compare_category].keys()),
            format_func=lambda x: f'Sample {x}',
            key='compare_sample'
        )
        
        st.markdown('---')
        st.markdown('### Thickness Parameters')
        
        fit_lipid = st.slider(
            'Lipid (nm)',
            10, 200, 80,
            key='fit_lipid'
        )
        
        fit_aqueous = st.slider(
            'Aqueous (nm)',
            500, 5000, 2337,
            step=25,
            key='fit_aqueous'
        )
        
        fit_mucus = st.slider(
            'Mucus (nm)',
            100, 1000, 500,
            step=25,
            key='fit_mucus'
        )
        
        st.markdown('---')
        auto_fit = st.button('🔍 Auto-Fit (Simple Grid)', use_container_width=True)
    
    with col2:
        sample_info = samples[compare_category][compare_sample]
        
        if sample_info['measured'] and sample_info['bestfit']:
            # Load measured and bestfit spectra
            measured_wl, measured_refl = load_measured_spectrum(sample_info['measured'])
            bestfit_wl, bestfit_refl = load_bestfit_spectrum(sample_info['bestfit'])
            
            # Common wavelength range
            wl_min = max(measured_wl.min(), bestfit_wl.min(), 600)
            wl_max = min(measured_wl.max(), bestfit_wl.max(), 1100)
            
            common_wavelengths = np.linspace(wl_min, wl_max, 500)
            measured_interp = np.interp(common_wavelengths, measured_wl, measured_refl)
            bestfit_interp = np.interp(common_wavelengths, bestfit_wl, bestfit_refl)
            
            # Calculate pyElli theoretical
            lipid_df = load_material_data(materials['lipid_05-02621extrapolated'])
            water_df = load_material_data(materials['water_Bashkatov1353extrapolated'])
            mucus_df = load_material_data(materials['struma_Bashkatov140extrapolated'])
            
            lipid_n, lipid_k = get_nk(lipid_df, common_wavelengths)
            water_n, water_k = get_nk(water_df, common_wavelengths)
            mucus_n, mucus_k = get_nk(mucus_df, common_wavelengths)
            
            layers = [
                (lipid_n, lipid_k, fit_lipid),
                (water_n, water_k, fit_aqueous),
                (mucus_n, mucus_k, fit_mucus),
            ]
            
            pyelli_refl = transfer_matrix_reflectance(common_wavelengths, layers)
            
            # Scale pyElli to match measured amplitude
            scale_factor = np.mean(measured_interp) / np.mean(pyelli_refl)
            pyelli_scaled = pyelli_refl * scale_factor
            
            # Calculate metrics
            residual_lta = calculate_residual(measured_interp, bestfit_interp)
            residual_pyelli = calculate_residual(measured_interp, pyelli_scaled)
            corr_lta = calculate_correlation(measured_interp, bestfit_interp)
            corr_pyelli = calculate_correlation(measured_interp, pyelli_scaled)
            
            # Create comparison figure
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.65, 0.35],
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=['Spectrum Comparison', 'Residuals']
            )
            
            # Spectra
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths, y=measured_interp,
                    mode='lines', name='Measured',
                    line=dict(color=COLORS['measured'], width=2.5)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths, y=bestfit_interp,
                    mode='lines', name='LTA BestFit',
                    line=dict(color=COLORS['bestfit'], width=2.5, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths, y=pyelli_scaled,
                    mode='lines', name='TMM Theoretical',
                    line=dict(color=COLORS['theoretical'], width=2.5, dash='dot')
                ),
                row=1, col=1
            )
            
            # Residuals
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths, y=measured_interp - bestfit_interp,
                    mode='lines', name='LTA Residual',
                    line=dict(color=COLORS['bestfit'], width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(244, 114, 182, 0.1)'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths, y=measured_interp - pyelli_scaled,
                    mode='lines', name='TMM Residual',
                    line=dict(color=COLORS['theoretical'], width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(167, 139, 250, 0.1)'
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
            
            fig.update_xaxes(title_text='Wavelength (nm)', row=2, col=1, **PLOTLY_TEMPLATE['layout']['xaxis'])
            fig.update_yaxes(title_text='Reflectance', row=1, col=1, **PLOTLY_TEMPLATE['layout']['yaxis'])
            fig.update_yaxes(title_text='Residual', row=2, col=1, **PLOTLY_TEMPLATE['layout']['yaxis'])
            
            fig.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#ffffff',
                font=dict(family='DM Sans, sans-serif', color='#374151'),
                height=550,
                margin=dict(t=40, b=40, l=60, r=30),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor='#ffffff',
                    bordercolor='#e5e7eb',
                    font=dict(color='#374151')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics comparison
            st.markdown('### Fit Quality Metrics')
            
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric(
                    'LTA Residual',
                    f'{residual_lta:.6f}',
                    delta=None
                )
            
            with metric_cols[1]:
                st.metric(
                    'TMM Residual',
                    f'{residual_pyelli:.6f}',
                    delta=f'{(residual_pyelli - residual_lta) / residual_lta * 100:.1f}%' if residual_lta > 0 else None,
                    delta_color='inverse'
                )
            
            with metric_cols[2]:
                st.metric(
                    'LTA Correlation',
                    f'{corr_lta:.4f}'
                )
            
            with metric_cols[3]:
                st.metric(
                    'TMM Correlation',
                    f'{corr_pyelli:.4f}',
                    delta=f'{(corr_pyelli - corr_lta) * 100:.2f}%' if corr_lta > 0 else None
                )
            
            # Auto-fit result
            if auto_fit:
                st.markdown('### 🔍 Grid Search Results')
                with st.spinner('Running grid search...'):
                    best_residual = float('inf')
                    best_params = {}
                    
                    # Coarse grid search
                    for lipid in range(20, 150, 20):
                        for aqueous in range(1500, 4000, 250):
                            for mucus in range(200, 600, 100):
                                layers_test = [
                                    (lipid_n, lipid_k, lipid),
                                    (water_n, water_k, aqueous),
                                    (mucus_n, mucus_k, mucus),
                                ]
                                test_refl = transfer_matrix_reflectance(common_wavelengths, layers_test)
                                test_scaled = test_refl * (np.mean(measured_interp) / np.mean(test_refl))
                                test_residual = calculate_residual(measured_interp, test_scaled)
                                
                                if test_residual < best_residual:
                                    best_residual = test_residual
                                    best_params = {'lipid': lipid, 'aqueous': aqueous, 'mucus': mucus}
                    
                    st.success(f'''
                    **Best Parameters Found:**
                    - Lipid: {best_params['lipid']} nm
                    - Aqueous: {best_params['aqueous']} nm
                    - Mucus: {best_params['mucus']} nm
                    - Residual: {best_residual:.6f}
                    ''')


# =============================================================================
# Tab 5: Integration Guide
# =============================================================================

with tabs[4]:
    st.markdown('''
    <div style="margin-bottom: 24px;">
        <h2>📚 Integration Guide</h2>
        <p style="color: #94a3b8;">Comprehensive guide for integrating pyElli into the AdOM-TFI workflow.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    ### What is PyElli?
    
    **pyElli** is an open-source Python library for spectroscopic ellipsometry and
    thin film optics calculations. Key features include:
    
    - 🔢 **Transfer Matrix Method (TMM)** - Standard approach for multi-layer thin film calculations
    - 🌈 **Dispersion Models** - Cauchy, Sellmeier, Lorentz, Drude, and tabulated data support
    - 📐 **Ellipsometry Fitting** - Psi/Delta fitting with various optimizers
    - 🔄 **Wavelength-by-wavelength** or **Global** fitting modes
    
    ### Applicability to AdOM-TFI
    
    Based on the codebase analysis, here's how pyElli could enhance the workflow:
    ''')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        #### ✅ High-Value Applications
        
        1. **Replace/Validate SpAnalizer.dll**
           - Open-source alternative to proprietary DLL
           - Cross-platform (no Windows dependency)
           - Full control over calculation parameters
        
        2. **Dispersion Model Flexibility**
           - Use tabulated n,k data directly
           - Fit Cauchy/Sellmeier models to material data
           - Model temperature-dependent properties
        
        3. **Advanced Fitting Algorithms**
           - scipy.optimize integration
           - Global optimization methods
           - Constrained fitting
        
        4. **Uncertainty Quantification**
           - Parameter covariance estimation
           - Confidence intervals on thicknesses
        ''')
    
    with col2:
        st.markdown('''
        #### 🔄 Integration Considerations
        
        1. **Material Data Conversion**
           ```python
           # Convert existing CSV to pyElli format
           from elli.dispersions import TableIndex
           
           # Load your material data
           mat = TableIndex.from_csv('lipid.csv')
           ```
        
        2. **Structure Building**
           ```python
           import elli
           
           # Build tear film structure
           structure = elli.Structure(
               elli.AIR,
               elli.Layer(lipid_mat, d_lipid),
               elli.Layer(aqueous_mat, d_aqueous),
               elli.Layer(mucus_mat, d_mucus),
               substrate
           )
           ```
        
        3. **Reflectance Calculation**
           ```python
           # Calculate reflectance
           result = structure.evaluate(
               wavelengths, 
               angle=0,
               polarization='s'
           )
           R = result.R_s
           ```
        ''')
    
    st.markdown('---')
    
    st.markdown('''
    ### Recommended Integration Path
    
    ```mermaid
    graph LR
        A[Current: SpAnalizer.dll] --> B[Phase 1: Validation]
        B --> C[Phase 2: Parallel Implementation]
        C --> D[Phase 3: Full Migration]
        
        B --> B1[Compare TMM vs DLL output]
        B --> B2[Verify material interpolation]
        B --> B3[Benchmark performance]
        
        C --> C1[pyElli grid search module]
        C --> C2[Side-by-side results]
        C --> C3[Regression testing]
        
        D --> D1[Replace DLL dependency]
        D --> D2[Extend with new features]
        D --> D3[Integrate fitting algorithms]
    ```
    ''')
    
    st.markdown('---')
    
    st.markdown('### Code Example: Full Tear Film Analysis')
    
    st.code('''
import numpy as np
import elli
from elli.dispersions import TableIndex, Cauchy
from scipy.optimize import minimize

# 1. Load material data
lipid = TableIndex.from_file('data/Materials/lipid_extrapolated.csv')
water = TableIndex.from_file('data/Materials/water_extrapolated.csv')  
mucus = TableIndex.from_file('data/Materials/mucus_extrapolated.csv')

# 2. Define structure function
def build_tear_film(d_lipid, d_aqueous, d_mucus):
    return elli.Structure(
        elli.AIR,
        elli.Layer(lipid, d_lipid),
        elli.Layer(water, d_aqueous),
        elli.Layer(mucus, d_mucus),
        elli.IsotropicMaterial(n=1.38)  # Simplified eye substrate
    )

# 3. Define objective function
def residual(params, wavelengths, measured_R):
    d_lipid, d_aqueous, d_mucus = params
    structure = build_tear_film(d_lipid, d_aqueous, d_mucus)
    
    result = structure.evaluate(wavelengths, angle=0)
    theoretical_R = result.R_s
    
    return np.sum((measured_R - theoretical_R)**2)

# 4. Run optimization
initial_guess = [80, 2500, 500]  # nm
bounds = [(20, 200), (500, 5000), (100, 1000)]

result = minimize(
    residual, 
    initial_guess,
    args=(wavelengths, measured_reflectance),
    bounds=bounds,
    method='L-BFGS-B'
)

best_lipid, best_aqueous, best_mucus = result.x
print(f"Best fit: Lipid={best_lipid:.1f}nm, Aqueous={best_aqueous:.1f}nm, Mucus={best_mucus:.1f}nm")
''', language='python')
    
    st.markdown('---')
    
    st.info('''
    **Next Steps for Team Discussion:**
    1. Run validation tests comparing pyElli TMM vs current SpAnalizer.dll output
    2. Quantify computational performance differences  
    3. Evaluate fitting convergence on good_fit vs bad_fit samples
    4. Decide on phased integration timeline
    ''')


# =============================================================================
# Footer
# =============================================================================

st.markdown('''
<div class="footer">
    <p style="font-size: 0.85rem; margin-bottom: 4px; color: #64748b;">
        <span style="color: #1e40af; font-weight: 600;">PyElli Exploration Tool</span> | AdOM Tear Film Interferometry Project
    </p>
    <p style="font-size: 0.75rem; color: #94a3b8;">For research and development purposes only</p>
</div>
''', unsafe_allow_html=True)

