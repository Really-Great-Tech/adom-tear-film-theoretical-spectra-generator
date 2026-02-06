"""Helper functions for displaying quality metrics in Streamlit apps."""

import streamlit as st
import plotly.graph_objects as go
from analysis.quality_metrics import assess_spectrum_quality


def display_quality_metrics_card(
    wavelengths,
    reflectance,
    fitted_spectrum=None,
    prominence=0.0001,
    config=None,
):
    """Display comprehensive quality metrics in a Streamlit card.

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
    with st.expander("📊 Quality Metrics Details", expanded=True):
        cols = st.columns(2)

        metric_idx = 0
        for name, result in report.metrics.items():
            col = cols[metric_idx % 2]
            metric_idx += 1

            with col:
                status_color = quality_colors.get(result.status, "#64748b")
                status_icon = quality_icons.get(result.status, "")

                st.markdown(
                    f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 600; color: #1e40af; margin-bottom: 8px;">
                        <span>{status_icon}</span>
                        <span>{name.replace("_", " ").title()}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b;">
                        <div style="margin-bottom: 4px;">Value: <span style="font-family: monospace; font-weight: 600; color: {status_color}; text-shadow: 0 0 1px {status_color}33;">{result.value:.4f}</span></div>
                        <div>Threshold: <span style="font-family: monospace;">{result.threshold:.4f}</span></div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Show key details
                if name == "snr":
                    st.caption(
                        f"SNR: {result.details.get('snr', 0):.2f} ({result.status})"
                    )
                elif name == "peak_quality":
                    st.caption(
                        f"Peaks: {int(result.details.get('peak_count', 0))}, Prom CV: {result.details.get('prominence_cv', 0):.3f}"
                    )
                elif name == "signal_integrity":
                    st.caption(
                        f"Range: {result.details.get('dynamic_range', 0):.4f}, Integrity: {100 - result.details.get('baseline_drift_percent', 0):.1f}%"
                    )
                elif name == "spectral_completeness":
                    st.caption(
                        f"Span: {result.details.get('wavelength_span_nm', 0):.1f} nm"
                    )

    # Display warnings and failures
    if report.warnings:
        with st.expander("⚠️ Warnings", expanded=False):
            for warning in report.warnings:
                st.warning(warning)

    if report.failures:
        with st.expander("❌ Failures", expanded=True):
            for failure in report.failures:
                st.error(failure)

    return report


def display_sliding_window_snr_chart(sw_data):
    """Display the sliding window SNR chart with intensity shading."""
    if not sw_data:
        return

    sw_window = sw_data.get("window_nm", 50.0)

    fig_sw = go.Figure()

    # 1. The Intensity (Bar Chart Underlay)
    fig_sw.add_trace(
        go.Bar(
            x=sw_data["centers"],
            y=sw_data["snr_values"],
            marker=dict(
                color=sw_data["snr_values"],
                colorscale="Blues",
                cmin=0,
                showscale=True,
                colorbar=dict(
                    title="SNR Intensity",
                    thickness=15,
                    len=0.8,
                    y=0.5,
                    x=1.05,
                ),
            ),
            width=sw_data.get("stride_nm", 25.0),
            opacity=0.5,
            showlegend=False,
            name="Intensity",
            hoverinfo="skip",
        )
    )

    # 2. The Line (Trend overlay)
    fig_sw.add_trace(
        go.Scatter(
            x=sw_data["centers"],
            y=sw_data["snr_values"],
            mode="lines+markers",
            name="SNR Profile",
            line=dict(color="#1e40af", width=2),
            marker=dict(size=4, color="#1e40af"),
            hovertemplate=(
                "<b>Wavelength Range</b>: %{customdata[0]:.1f} - %{customdata[1]:.1f} nm<br>"
                + "<b>SNR</b>: %{y:.1f}<br>"
                + "<extra></extra>"
            ),
            customdata=sw_data["ranges"],
        )
    )

    # Add threshold line (Robust Gate: 150.0 is Marginal)
    fig_sw.add_hline(
        y=150.0,
        line_dash="dash",
        line_color="#dc2626",
        line_width=2,
        annotation_text="Threshold (150.0)",
        annotation_position="bottom right",
    )

    fig_sw.update_layout(
        title=f"SNR vs Wavelength ({sw_window:.0f}nm Sliding Windows)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="SNR Ratio (Variance)",
        template="plotly_white",
        height=450,
        hovermode="closest",
        bargap=0,  # Makes bars touch for continuous look
        margin=dict(l=20, r=80, t=50, b=20),
    )

    st.plotly_chart(fig_sw, use_container_width=True)
