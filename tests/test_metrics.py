import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from analysis import (
    prepare_measurement,
    prepare_theoretical_spectrum,
    peak_count_score,
    peak_delta_score,
    phase_overlap_score,
    composite_score,
)


@pytest.fixture
def analysis_cfg():
    return {
        "detrending": {"default_cutoff_frequency": 0.01, "filter_order": 3},
        "peak_detection": {"default_prominence": 0.05, "min_height": None},
        "metrics": {
            "peak_count": {"wavelength_tolerance_nm": 5.0},
            "peak_delta": {"tolerance_nm": 5.0, "tau_nm": 10.0, "penalty_unpaired": 0.0},
            "phase_overlap": {"resample_points": 256, "window": "hann"},
            "composite": {
                "weights": {"peak_count": 0.4, "peak_delta": 0.4, "phase_overlap": 0.2}
            },
        },
    }


@pytest.fixture
def measurement_df():
    wavelengths = np.linspace(500, 650, 300)
    reflectance = 0.2 * np.sin(np.linspace(0, 8 * np.pi, wavelengths.size)) + 0.5
    return pd.DataFrame({"wavelength": wavelengths, "reflectance": reflectance})


def test_metrics_identical_spectrum(analysis_cfg, measurement_df):
    measurement_features = prepare_measurement(measurement_df, analysis_cfg)
    theoretical = prepare_theoretical_spectrum(
        measurement_df["wavelength"].to_numpy(),
        measurement_df["reflectance"].to_numpy(),
        measurement_features,
        analysis_cfg,
    )

    count_result = peak_count_score(
        measurement_features, theoretical, tolerance_nm=analysis_cfg["metrics"]["peak_count"]["wavelength_tolerance_nm"]
    )
    delta_result = peak_delta_score(
        measurement_features,
        theoretical,
        tolerance_nm=analysis_cfg["metrics"]["peak_delta"]["tolerance_nm"],
        tau_nm=analysis_cfg["metrics"]["peak_delta"]["tau_nm"],
        penalty_unpaired=analysis_cfg["metrics"]["peak_delta"]["penalty_unpaired"],
    )
    phase_result = phase_overlap_score(measurement_features, theoretical)

    assert count_result.score == pytest.approx(1.0, rel=1e-4)
    assert delta_result.score == pytest.approx(1.0, rel=1e-4)
    assert phase_result.score == pytest.approx(1.0, rel=1e-4)

    composite = composite_score(
        {
            "peak_count": count_result.score,
            "peak_delta": delta_result.score,
            "phase_overlap": phase_result.score,
        },
        analysis_cfg["metrics"]["composite"]["weights"],
    )
    assert composite == pytest.approx(1.0, rel=1e-4)


def test_peak_delta_penalty_for_shift(analysis_cfg, measurement_df):
    measurement_features = prepare_measurement(measurement_df, analysis_cfg)
    shifted_reflectance = np.roll(measurement_df["reflectance"].to_numpy(), 5)
    theoretical = prepare_theoretical_spectrum(
        measurement_df["wavelength"].to_numpy(),
        shifted_reflectance,
        measurement_features,
        analysis_cfg,
    )

    delta_result = peak_delta_score(
        measurement_features,
        theoretical,
        tolerance_nm=analysis_cfg["metrics"]["peak_delta"]["tolerance_nm"],
        tau_nm=analysis_cfg["metrics"]["peak_delta"]["tau_nm"],
        penalty_unpaired=analysis_cfg["metrics"]["peak_delta"]["penalty_unpaired"],
    )

    assert delta_result.score < 1.0
    assert delta_result.diagnostics["mean_delta_nm"] > 0
