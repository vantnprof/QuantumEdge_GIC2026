import numpy as np

from quantumedge.evaluation.metrics import compute_all, regime_accuracy


def test_compute_all_returns_expected_metrics():
    values = compute_all(
        np.array([1.0, 2.0, 4.0]),
        np.array([1.0, 2.5, 3.5]),
        include_qlike=False,
    )

    assert np.isclose(values["rmse"], np.sqrt((0.0**2 + 0.5**2 + 0.5**2) / 3))
    assert np.isclose(values["mae"], (0.0 + 0.5 + 0.5) / 3)
    assert "qlike" not in values


def test_regime_accuracy_uses_thresholds():
    true_regimes = np.array([0, 1, 2, 2])
    pred_rv = np.array([0.1, 0.3, 0.8, 0.9])

    assert regime_accuracy(true_regimes, pred_rv, thresholds=(0.2, 0.7)) == 1.0
