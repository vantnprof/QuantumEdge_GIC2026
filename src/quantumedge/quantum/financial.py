"""Notebook-style financial QRC inputs and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from quantumedge.models.hmm_regimes import causal_regime_posteriors, fit_hmm, predict_regimes
from quantumedge.quantum.features import (
    entanglement_energies_from_state,
    pauli_z_features_from_state,
)


ANGLE_RANGE = (-np.pi / 2, np.pi / 2)
SHORT_FINANCIAL_COLS = ("log_rv_d", "log_ret", "log_gk")
LONG_FINANCIAL_COLS = ("log_rv_w", "log_rv_m", "vix", "vix_rv_spread")


@dataclass(frozen=True)
class FinancialQRCInputs:
    dataset: str
    frame: pd.DataFrame
    short_angles: np.ndarray
    long_angles: np.ndarray
    target_rv: np.ndarray
    test_returns: np.ndarray
    true_test_regimes: np.ndarray
    train_regimes: np.ndarray
    regime_posteriors: np.ndarray
    test_posteriors: np.ndarray
    angle_scaler: MinMaxScaler
    hmm_thresholds: tuple[float, float]
    n_train: int
    n_test: int


def _fit_feature_scalers(train_df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, MinMaxScaler]:
    scalers = {}
    for column in columns:
        scaler = MinMaxScaler(feature_range=ANGLE_RANGE)
        scaler.fit(train_df[[column]])
        scalers[column] = scaler
    return scalers


def _encode_frame(frame: pd.DataFrame, columns: tuple[str, ...], scalers: dict[str, MinMaxScaler]) -> np.ndarray:
    out = np.zeros((len(frame), len(columns)), dtype=float)
    for idx, column in enumerate(columns):
        values = scalers[column].transform(frame[[column]]).ravel()
        out[:, idx] = np.clip(values, ANGLE_RANGE[0], ANGLE_RANGE[1])
    return out


def build_financial_qrc_inputs(dataset: str, n_train: int, n_test: int) -> FinancialQRCInputs:
    """Prepare multi-feature financial QRC inputs using the notebook setup."""
    from quantumedge.data.loaders import load_all
    from quantumedge.features.builders import prepare_all

    name = dataset.lower().replace("-", "_")
    if name not in {"sp500_rv", "oxford_man_rv"}:
        raise ValueError(f"financial QRC inputs support realized-variance datasets, got {dataset}")

    prepared = prepare_all(load_all())
    frame = prepared["financial"].copy()
    required = n_train + n_test
    if len(frame) < required:
        raise ValueError(f"{dataset} has {len(frame)} usable rows, but {required} are required")
    frame = frame.iloc[:required].copy()
    train_frame = frame.iloc[:n_train].copy()

    short_scalers = _fit_feature_scalers(train_frame, SHORT_FINANCIAL_COLS)
    long_scalers = _fit_feature_scalers(train_frame, LONG_FINANCIAL_COLS)
    short_angles = _encode_frame(frame, SHORT_FINANCIAL_COLS, short_scalers)
    long_angles = _encode_frame(frame, LONG_FINANCIAL_COLS, long_scalers)

    # Append one target-only row so the feature matrix has exactly
    # n_train + n_test one-step targets, matching the classical split.
    terminal_short = short_angles[-1:].copy()
    terminal_long = long_angles[-1:].copy()
    terminal_log_rv = np.log(max(float(frame["target_rv"].iloc[-1]), 1e-10))
    terminal_log_rv_frame = pd.DataFrame({"log_rv_d": [terminal_log_rv]})
    terminal_short[0, 0] = np.clip(
        short_scalers["log_rv_d"].transform(terminal_log_rv_frame)[0, 0],
        ANGLE_RANGE[0],
        ANGLE_RANGE[1],
    )
    short_angles = np.vstack([short_angles, terminal_short])
    long_angles = np.vstack([long_angles, terminal_long])

    financial_split = {"train": train_frame, "test": frame.iloc[n_train:].copy()}
    hmm_model, state_map, thresholds = fit_hmm(financial_split)
    posteriors = causal_regime_posteriors(hmm_model, state_map, frame["rv"].to_numpy(dtype=float))

    target_rv = frame["target_rv"].to_numpy(dtype=float)[n_train:n_train + n_test]
    return FinancialQRCInputs(
        dataset=name,
        frame=frame,
        short_angles=short_angles,
        long_angles=long_angles,
        target_rv=target_rv,
        test_returns=frame["log_ret"].to_numpy(dtype=float)[n_train:n_train + n_test],
        true_test_regimes=predict_regimes(hmm_model, state_map, target_rv),
        train_regimes=predict_regimes(hmm_model, state_map, train_frame["rv"].to_numpy(dtype=float)),
        regime_posteriors=posteriors,
        test_posteriors=posteriors[n_train:n_train + n_test],
        angle_scaler=short_scalers["log_rv_d"],
        hmm_thresholds=thresholds,
        n_train=n_train,
        n_test=n_test,
    )


def angle_predictions_to_rv(angle_scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """Convert log-RV angle predictions to strictly positive RV values."""
    clipped = np.clip(np.asarray(values, dtype=float), ANGLE_RANGE[0], ANGLE_RANGE[1])
    log_rv = angle_scaler.inverse_transform(clipped.reshape(-1, 1)).ravel()
    return np.exp(log_rv)


def extract_financial_statevector_features(
    inputs: FinancialQRCInputs,
    short_node,
    long_node,
    n_qubits: int,
    n_ent: int | None = None,
    label: str = "financial",
    progress_every: int = 250,
) -> dict[str, np.ndarray]:
    """Drive notebook-style short/long reservoirs over financial angle matrices."""
    import time

    n_ent = n_ent if n_ent is not None else min(2 ** (n_qubits // 2), 8)
    total_rows = inputs.n_train + inputs.n_test
    dual_pauli_rows = []
    dual_ent_rows = []
    dual_ent_regime_rows = []
    single_long_ent_rows = []
    start = time.perf_counter()

    for idx in range(total_rows):
        state_short = short_node(inputs.short_angles[idx])
        state_long = long_node(inputs.long_angles[idx])

        pauli_short = pauli_z_features_from_state(state_short, n_qubits)
        pauli_long = pauli_z_features_from_state(state_long, n_qubits)
        ent_short = entanglement_energies_from_state(state_short, n_qubits, n_ent)
        ent_long = entanglement_energies_from_state(state_long, n_qubits, n_ent)
        feedback = np.array([float(inputs.short_angles[idx, 0])])
        # Use causal posteriors for train and test features. They are computed
        # only from observations up to each time step.
        regime = inputs.regime_posteriors[idx]

        dual_pauli_rows.append(np.concatenate([pauli_short, pauli_long, feedback]))
        dual_ent_rows.append(np.concatenate([pauli_short, pauli_long, ent_short, ent_long, feedback]))
        dual_ent_regime_rows.append(
            np.concatenate([pauli_short, pauli_long, ent_short, ent_long, feedback, regime])
        )
        single_long_ent_rows.append(np.concatenate([pauli_long, ent_long, feedback]))

        if progress_every and ((idx + 1) % progress_every == 0 or (idx + 1) == total_rows):
            elapsed = time.perf_counter() - start
            print(f"{label:<15} {idx + 1:4d}/{total_rows} feature rows in {elapsed:6.1f}s")

    return {
        "dual_pauli": np.vstack(dual_pauli_rows),
        "dual_ent": np.vstack(dual_ent_rows),
        "dual_ent_regime": np.vstack(dual_ent_regime_rows),
        "single_long_ent": np.vstack(single_long_ent_rows),
        "target": inputs.short_angles[1:total_rows + 1, 0],
    }
