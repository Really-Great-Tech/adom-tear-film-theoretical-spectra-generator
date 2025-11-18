import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.analysis import (
    prepare_measurement,
    prepare_theoretical_spectrum,
    peak_count_score,
    peak_delta_score,
    phase_overlap_score,
    composite_score,
    residual_score,
    measurement_quality_score,
    temporal_continuity_score,
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


def test_residual_score_rewards_close_fit(analysis_cfg, measurement_df):
    measurement_features = prepare_measurement(measurement_df, analysis_cfg)
    theoretical_identical = prepare_theoretical_spectrum(
        measurement_df["wavelength"].to_numpy(),
        measurement_df["reflectance"].to_numpy(),
        measurement_features,
        analysis_cfg,
    )
    noisy_reflectance = measurement_df["reflectance"].to_numpy() + 0.05
    theoretical_noisy = prepare_theoretical_spectrum(
        measurement_df["wavelength"].to_numpy(),
        noisy_reflectance,
        measurement_features,
        analysis_cfg,
    )

    good = residual_score(measurement_features, theoretical_identical, tau_rmse=0.01, max_rmse=1.0)
    bad = residual_score(measurement_features, theoretical_noisy, tau_rmse=0.01, max_rmse=1.0)

    assert good.score > bad.score
    assert good.diagnostics["rmse"] < bad.diagnostics["rmse"]


def test_measurement_quality_score_flags_low_amplitude(analysis_cfg, measurement_df):
    measurement_features = prepare_measurement(measurement_df, analysis_cfg)
    score, failures = measurement_quality_score(
        measurement_features,
        min_peaks=1,
        min_signal_amplitude=0.01,
        min_wavelength_span_nm=10,
    )
    assert score.score == pytest.approx(1.0)
    assert not failures

    flat_df = measurement_df.copy()
    flat_df["reflectance"] = 0.5
    flat_features = prepare_measurement(flat_df, analysis_cfg)
    low_score, low_failures = measurement_quality_score(
        flat_features,
        min_peaks=1,
        min_signal_amplitude=0.01,
    )
    assert low_score.score < 1.0
    assert "min_signal_amplitude" in low_failures


def test_temporal_continuity_penalizes_jumps():
    close = temporal_continuity_score(
        {"lipid_nm": 80.0, "aqueous_nm": 1000.0, "roughness_A": 2000.0},
        {"lipid_nm": 82.0, "aqueous_nm": 995.0, "roughness_A": 2010.0},
        tau_lipid_nm=10.0,
        tau_aqueous_nm=50.0,
        tau_roughness_A=100.0,
    )
    far = temporal_continuity_score(
        {"lipid_nm": 80.0, "aqueous_nm": 1000.0, "roughness_A": 2000.0},
        {"lipid_nm": 140.0, "aqueous_nm": 800.0, "roughness_A": 2600.0},
        tau_lipid_nm=10.0,
        tau_aqueous_nm=50.0,
        tau_roughness_A=100.0,
    )
    assert close.score > far.score
