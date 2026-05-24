import numpy as np
import pytest

from quantumedge.quantum.features import extract_statevector_features, make_statevector_qnode
from quantumedge.quantum.readouts import train_ridge_readout
from quantumedge.quantum.reservoirs import make_reservoir_pair
from quantumedge.quantum.series import normalize_to_angles


def test_angle_normalization_uses_training_segment():
    raw = np.linspace(-1.0, 2.0, 20)
    raw[-1] = 10.0

    angles, _ = normalize_to_angles(raw, n_train=10)

    assert np.isclose(angles[:11].min(), -np.pi / 2)
    assert np.isclose(angles[:11].max(), np.pi / 2)
    assert angles[-1] == np.pi / 2


def test_statevector_qrc_feature_shapes_and_readout():
    pytest.importorskip("pennylane")

    short, long = make_reservoir_pair(
        n_qubits=3,
        short_trotter_depth=1,
        long_trotter_depth=2,
    )
    angles = np.linspace(-0.5, 0.5, 13)
    short_node = make_statevector_qnode(short, device_name="default.qubit")
    long_node = make_statevector_qnode(long, device_name="default.qubit")

    features = extract_statevector_features(
        angles,
        short_node,
        long_node,
        n_qubits=3,
        n_ent=2,
        progress_every=0,
    )

    assert features["dual_pauli"].shape == (12, 11)
    assert features["dual_ent"].shape == (12, 15)
    assert features["single_long_ent"].shape == (12, 8)
    assert features["target"].shape == (12,)

    result = train_ridge_readout(
        features["dual_pauli"],
        features["target"],
        n_train=8,
        n_test=4,
    )
    assert result["y_pred"].shape == (4,)
    assert result["rmse"] >= 0.0
