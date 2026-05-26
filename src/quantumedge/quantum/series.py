"""Time-series generators and angle encoders for QRC experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler

from quantumedge.quantum.reservoirs import DEFAULT_N_TOTAL, DEFAULT_N_TRAIN, SEED


def mackey_glass(
    n_steps: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: int = 10,
    dt: float = 1.0,
    x0: float = 1.2,
) -> np.ndarray:
    """Generate a Mackey-Glass delay-differential series with Euler stepping."""
    delay = int(tau / dt)
    history = delay + 1
    x = np.empty(n_steps + history, dtype=float)
    x[:history] = x0
    for t in range(history, n_steps + history):
        x_tau = x[t - delay]
        dx = beta * x_tau / (1.0 + x_tau ** exponent) - gamma * x[t - 1]
        x[t] = x[t - 1] + dt * dx
    return x[history:]


def lorenz_x_series(
    n_steps: int,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.02,
) -> np.ndarray:
    """Integrate the Lorenz system and return the x-coordinate."""

    def rhs(_t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_eval = np.arange(n_steps) * dt
    solution = solve_ivp(
        rhs,
        t_span=(t_eval[0], t_eval[-1]),
        y0=[1.0, 1.0, 1.0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Lorenz integration failed: {solution.message}")
    return solution.y[0]


def discard_transient(
    series: np.ndarray,
    transient_steps: int,
    n_total: int = DEFAULT_N_TOTAL,
) -> np.ndarray:
    """Drop the initial transient and keep exactly `n_total` observations."""
    usable = np.asarray(series[transient_steps:transient_steps + n_total], dtype=float)
    if len(usable) != n_total:
        raise ValueError(f"Expected {n_total} observations after transient, got {len(usable)}")
    return usable


def normalize_to_angles(
    raw_series: np.ndarray,
    n_train: int = DEFAULT_N_TRAIN,
    feature_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
) -> tuple[np.ndarray, MinMaxScaler]:
    """Fit an angle scaler on the training segment, then transform all values."""
    raw_series = np.asarray(raw_series, dtype=float)
    if len(raw_series) <= n_train:
        raise ValueError("raw_series must contain at least n_train + 1 observations")

    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(raw_series[:n_train + 1].reshape(-1, 1))
    angles = scaler.transform(raw_series.reshape(-1, 1)).ravel()
    angles = np.clip(angles, feature_range[0], feature_range[1])
    return angles, scaler


def build_mackey_glass_raw(n_total: int = DEFAULT_N_TOTAL) -> np.ndarray:
    """Build the Mackey-Glass benchmark series used in the QRC prototype."""
    np.random.seed(SEED)
    return discard_transient(mackey_glass(n_total + 150), transient_steps=150, n_total=n_total)


def build_lorenz_raw(n_total: int = DEFAULT_N_TOTAL) -> np.ndarray:
    """Build the Lorenz x-coordinate benchmark used in the QRC prototype."""
    np.random.seed(SEED)
    return discard_transient(lorenz_x_series(n_total + 300), transient_steps=300, n_total=n_total)


def build_chaotic_angles(
    dataset: str,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = 500,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Return normalized angle series for the supported chaotic benchmarks."""
    n_total = n_train + n_test + 1
    name = dataset.lower().replace("-", "_")
    if name == "mackey_glass":
        raw = build_mackey_glass_raw(n_total=n_total)
    elif name == "lorenz":
        raw = build_lorenz_raw(n_total=n_total)
    else:
        raise ValueError(f"unsupported chaotic QRC dataset: {dataset}")
    return normalize_to_angles(raw, n_train=n_train)


def project_chaotic_series_from_prepared(dataset: str, prepared: dict) -> np.ndarray:
    """Return the exact normalized target sequence used by classical chaotic splits."""
    name = dataset.lower().replace("-", "_")
    if name == "mackey_glass":
        split = prepared["mackey_glass_split"]
    elif name == "lorenz":
        split = prepared["lorenz_split"]
    else:
        raise ValueError(f"unsupported project chaotic dataset: {dataset}")

    initial_feedback = float(split["X_train"][0, -1])
    return np.concatenate(
        [
            np.array([initial_feedback], dtype=float),
            np.asarray(split["y_train"], dtype=float),
            np.asarray(split["y_test"], dtype=float),
        ]
    )


def build_project_chaotic_angles(
    dataset: str,
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Build QRC angles from the same chaotic split targets as the baselines."""
    from quantumedge.data.loaders import load_all
    from quantumedge.features.builders import prepare_all

    prepared = prepare_all(load_all())
    raw = project_chaotic_series_from_prepared(dataset, prepared)
    required = n_train + n_test + 1
    if len(raw) < required:
        raise ValueError(f"{dataset} has {len(raw)} aligned rows, but {required} are required")
    return normalize_to_angles(raw[:required], n_train=n_train)


def financial_series_from_prepared(dataset: str, prepared: dict) -> np.ndarray:
    """Extract the scalar financial series proposed for QRC angle encoding."""
    name = dataset.lower()
    if name in {"sp500_rv", "oxford_man_rv"}:
        financial = prepared["financial"]
        rv = financial["rv"].to_numpy(dtype=float)
        final_target = float(financial["target_rv"].iloc[-1])
        rv = np.concatenate([rv, [final_target]])
        return np.log(np.maximum(rv, 1e-10))
    if name == "vix":
        vix_df = prepared["vix_df"]
        vix = vix_df["vix"].to_numpy(dtype=float)
        final_target = float(vix_df["target_vix"].iloc[-1])
        vix = np.concatenate([vix, [final_target]])
        return np.log(np.maximum(vix, 1e-10))
    raise ValueError(f"unsupported financial QRC dataset: {dataset}")


def build_financial_angles(
    dataset: str,
    n_train: int | None = None,
    n_test: int | None = None,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Build QRC angles from the same prepared financial data as the baselines."""
    from quantumedge.data.loaders import load_all
    from quantumedge.features.builders import prepare_all

    data = load_all()
    prepared = prepare_all(data)
    raw = financial_series_from_prepared(dataset, prepared)

    if n_train is None:
        n_train = int(len(raw) * 0.80)
    if n_test is not None:
        required = n_train + n_test + 1
        if len(raw) < required:
            raise ValueError(
                f"{dataset} has {len(raw)} rows, but {required} are required"
            )
        raw = raw[:required]

    return normalize_to_angles(raw, n_train=n_train)


def build_angle_series(
    dataset: str,
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Build an angle series for chaotic or financial QRC datasets."""
    name = dataset.lower().replace("-", "_")
    if name in {"mackey_glass", "lorenz"}:
        return build_chaotic_angles(name, n_train=n_train, n_test=n_test)
    if name in {"sp500_rv", "oxford_man_rv", "vix"}:
        return build_financial_angles(name, n_train=n_train, n_test=n_test)
    raise ValueError(f"unsupported QRC dataset: {dataset}")


def normalization_summary(dataset: str, raw: np.ndarray, angles: np.ndarray) -> pd.DataFrame:
    """Small tabular diagnostic for reports and notebooks."""
    return pd.DataFrame(
        {
            "dataset": [dataset],
            "raw_min": [float(np.min(raw))],
            "raw_max": [float(np.max(raw))],
            "angle_min": [float(np.min(angles))],
            "angle_max": [float(np.max(angles))],
        }
    )
