"""Helper functions for displaying quality metrics in Streamlit apps."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.quality_metrics import assess_spectrum_quality, SpectrumQualityReport


def display_quality_metrics_card(
    wavelengths,
    reflectance,
    fitted_spectrum=None,
    prominence=0.0001,
    config=None,
):
    """Display quality metrics in a Streamlit card.

    Args:
        wavelengths: Wavelength array in nm
        reflectance: Reflectance values
        fitted_spectrum: Optional fitted/theoretical spectrum
        prominence: Peak detection prominence
        config: Optional quality metrics configuration
    """
    # Assess quality
    report = assess_spectrum_quality(
        wavelengths,
        reflectance,
        fitted_spectrum=fitted_spectrum,
        prominence=prominence,
        config=config,
    )

    # Display overall quality with color coding
    quality_colors = {
        "Excellent": "#16a34a",  # green
        "Good": "#2563eb",  # blue
        "Marginal": "#d97706",  # orange
        "Reject": "#dc2626",  # red
    }

    quality_icons = {
        "Excellent": "✅",
        "Good": "✓",
        "Marginal": "⚠️",
        "Reject": "❌",
    }

    color = quality_colors.get(report.overall_quality, "#64748b")
    icon = quality_icons.get(report.overall_quality, "")

    st.markdown(
        f"""
    <div style="background: {color}15; border: 2px solid {color}; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                    Overall Quality
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {color};">
                    {icon} {report.overall_quality}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: #64748b;">
                    {"All checks passed" if report.passed_all_checks else f"{len(report.failures)} checks failed"}
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display metric details in expandable sections
    with st.expander("📊 Quality Metrics Details", expanded=False):
        cols = st.columns(2)

        metric_idx = 0
        for name, result in report.metrics.items():
            col = cols[metric_idx % 2]
            metric_idx += 1

            with col:
                # Use sub-metric status for cards
                sub_quality_colors = {
                    "Excellent": "#16a34a",  # green
                    "Good": "#2563eb",  # blue
                    "Marginal": "#d97706",  # orange
                    "Reject": "#dc2626",  # red
                }
                sub_quality_icons = {
                    "Excellent": "✅",
                    "Good": "✓",
                    "Marginal": "⚠️",
                    "Reject": "❌",
                }

                status_color = sub_quality_colors.get(result.status, "#64748b")
                status_icon = sub_quality_icons.get(result.status, "")

                st.markdown(
                    f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-weight: 600; color: #1e40af; margin-bottom: 8px;">
                        {status_icon} {name.replace("_", " ").title()}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b;">
                        Value: <span style="font-weight: 600; color: {status_color};">{result.value:.4f}</span>
                        <br>Status: <span style="color: {status_color}; font-weight: 600;">{result.status}</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Show key details
                if name == "snr":
                    st.caption(
                        f"Quality Level: {result.details.get('quality_level', 'N/A')}"
                    )
                    st.caption(f"SNR: {result.details.get('snr', 0):.2f}")
                elif name == "peak_quality":
                    st.caption(
                        f"Peak Count: {int(result.details.get('peak_count', 0))}"
                    )
                    st.caption(
                        f"Prominence CV: {result.details.get('prominence_cv', 0):.3f}"
                    )
                    st.caption(f"Spacing CV: {result.details.get('spacing_cv', 0):.3f}")
                elif name == "fit_quality":
                    st.caption(f"RMSE: {result.details.get('rmse', 0):.4f}")
                    st.caption(f"NRMSE: {result.details.get('nrmse_percent', 0):.2f}%")
                    st.caption(f"R²: {result.details.get('r_squared', 0):.4f}")
                elif name == "signal_integrity":
                    st.caption(
                        f"Dynamic Range: {result.details.get('dynamic_range', 0):.4f}"
                    )
                    st.caption(
                        f"Baseline Drift: {result.details.get('baseline_drift_percent', 0):.2f}%"
                    )
                elif name == "spectral_completeness":
                    st.caption(
                        f"Wavelength Span: {result.details.get('wavelength_span_nm', 0):.1f} nm"
                    )
                    st.caption(
                        f"Point Density: {result.details.get('point_density', 0):.2f} pts/nm"
                    )

    # Display warnings and failures
    if report.warnings:
        with st.expander("⚠️ Warnings", expanded=False):
            for warning in report.warnings:
                st.warning(warning)

    if report.failures:
        with st.expander("❌ Failures", expanded=len(report.failures) > 0):
            for failure in report.failures:
                st.error(failure)

    return report


def display_quality_metrics_compact(
    wavelengths,
    reflectance,
    fitted_spectrum=None,
    prominence=0.0001,
    config=None,
):
    """Display quality metrics in a compact format (for sidebar or small spaces).

    Args:
        wavelengths: Wavelength array in nm
        reflectance: Reflectance values
        fitted_spectrum: Optional fitted/theoretical spectrum
        prominence: Peak detection prominence
        config: Optional quality metrics configuration
    """
    # Assess quality
    report = assess_spectrum_quality(
        wavelengths,
        reflectance,
        fitted_spectrum=fitted_spectrum,
        prominence=prominence,
        config=config,
    )

    # Display compact metrics
    quality_colors = {
        "Excellent": "#16a34a",
        "Good": "#2563eb",
        "Marginal": "#d97706",
        "Reject": "#dc2626",
    }

    color = quality_colors.get(report.overall_quality, "#64748b")

    cols = st.columns([1, 2])
    with cols[0]:
        st.metric("Quality", report.overall_quality)
    with cols[1]:
        # Show key metrics
        snr_result = report.metrics.get("snr")
        if snr_result:
            st.metric(
                "SNR",
                f"{snr_result.value:.1f}",
                delta=f"{'✓' if snr_result.passed else '✗'}",
            )

    return report


__all__ = [
    "display_quality_metrics_card",
    "display_quality_metrics_compact",
]
