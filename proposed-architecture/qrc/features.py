"""Feature extraction: Pauli expectations, entanglement energies, full pipeline."""

import time
import numpy as np

from .config import N_ENT_FEATURES


# ---------------------------------------------------------------------------
# Single-statevector feature extractors
# ---------------------------------------------------------------------------

def pauli_features(statevector, n_qubits: int) -> np.ndarray:
    """
    Compute <Z_i> for i in 0..n-1 and <Z_i Z_{i+1}> for i in 0..n-2.
    Returns array of length 2*n_qubits - 1.
    """
    probs = np.abs(np.asarray(statevector, dtype=np.complex128)) ** 2
    tensor = probs.reshape((2,) * n_qubits)

    single_z = []
    for w in range(n_qubits):
        marginal = tensor.sum(axis=tuple(a for a in range(n_qubits) if a != w))
        single_z.append(float(marginal[0] - marginal[1]))

    pair_zz = []
    for w in range(n_qubits - 1):
        keep = (w, w + 1)
        marginal = tensor.sum(axis=tuple(a for a in range(n_qubits) if a not in keep))
        pair_zz.append(float(marginal[0, 0] + marginal[1, 1] - marginal[0, 1] - marginal[1, 0]))

    return np.array(single_z + pair_zz, dtype=float)


def entanglement_features(statevector, n_qubits: int, n_keep: int = N_ENT_FEATURES) -> np.ndarray:
    """
    Half-chain entanglement energies xi_k = -log(lambda_k) from the reduced
    density matrix of the left half of the chain.

    These encode multi-body quantum correlations that are provably inaccessible
    to classical reservoirs of the same dimension — the core quantum advantage claim.
    """
    n_left = n_qubits // 2
    n_right = n_qubits - n_left
    psi = np.asarray(statevector, dtype=np.complex128).reshape(2 ** n_left, 2 ** n_right)
    rho_left = psi @ psi.conj().T
    eigvals = np.linalg.eigvalsh(rho_left).real
    eigvals = np.sort(np.clip(eigvals, 1e-12, 1.0))[::-1]
    energies = -np.log(eigvals)
    if len(energies) < n_keep:
        energies = np.pad(energies, (0, n_keep - len(energies)), constant_values=0.0)
    return energies[:n_keep]


# ---------------------------------------------------------------------------
# Full pipeline extractor
# ---------------------------------------------------------------------------

def extract_financial_features(
    angle_matrix_short: np.ndarray,
    angle_matrix_long: np.ndarray,
    short_qnode,
    long_qnode,
    n_qubits: int,
    regime_posteriors=None,
    n_ent: int = N_ENT_FEATURES,
    label: str = "",
    progress_every: int = 200,
) -> dict:
    """
    Drive both reservoirs over the full financial angle series.

    Parameters
    ----------
    angle_matrix_short : (T+1, n_short_features) angle array
    angle_matrix_long  : (T+1, n_long_features) angle array
    short_qnode        : statevector QNode for short reservoir
    long_qnode         : statevector QNode for long reservoir
    n_qubits           : qubits per reservoir
    regime_posteriors  : (T+1, 3) HMM posteriors, or None
    n_ent              : number of entanglement energy features per reservoir
    label              : progress label
    progress_every     : print progress every N steps

    Returns
    -------
    dict with keys:
        dual_pauli       (T, 2*n_pauli + 1)
        dual_ent         (T, 2*(n_pauli + n_ent) + 1)
        dual_ent_regime  (T, 2*(n_pauli + n_ent) + 1 + 3)  [if posteriors given]
        single_long_ent  (T, n_pauli + n_ent + 1)
        target_angle     (T,)  normalised next-step rv_d angle
        ridge_alphas     {}    filled by readout module

    The feedback term (prev_rv_angle) is appended as a single scalar equal to
    the current-step rv_d angle, giving the readout an explicit autoregressive
    signal without any quantum parameters.
    """
    T = angle_matrix_short.shape[0] - 1
    dual_pauli_rows, dual_ent_rows, dual_ent_regime_rows, single_long_ent_rows = [], [], [], []
    t_start = time.perf_counter()

    for t in range(T):
        state_s = short_qnode(angle_matrix_short[t])
        state_l = long_qnode(angle_matrix_long[t])

        pauli_s = pauli_features(state_s, n_qubits)
        pauli_l = pauli_features(state_l, n_qubits)
        ent_s = entanglement_features(state_s, n_qubits, n_ent)
        ent_l = entanglement_features(state_l, n_qubits, n_ent)

        # Feedback: previous rv_d angle (first feature of short encoder)
        prev_rv = np.array([float(angle_matrix_short[t, 0])])

        dual_pauli_rows.append(np.concatenate([pauli_s, pauli_l, prev_rv]))
        dual_ent_rows.append(np.concatenate([pauli_s, pauli_l, ent_s, ent_l, prev_rv]))
        single_long_ent_rows.append(np.concatenate([pauli_l, ent_l, prev_rv]))

        if regime_posteriors is not None:
            dual_ent_regime_rows.append(
                np.concatenate([pauli_s, pauli_l, ent_s, ent_l, prev_rv,
                                regime_posteriors[t]])
            )

        if progress_every and ((t + 1) % progress_every == 0 or t + 1 == T):
            elapsed = time.perf_counter() - t_start
            print(f"{label:<20} {t + 1:5d}/{T}  {elapsed:7.1f}s")

    target_angle = angle_matrix_short[1:, 0]  # next-step rv_d in angle space

    out = {
        "dual_pauli": np.vstack(dual_pauli_rows),
        "dual_ent": np.vstack(dual_ent_rows),
        "single_long_ent": np.vstack(single_long_ent_rows),
        "target_angle": target_angle,
    }
    if regime_posteriors is not None:
        out["dual_ent_regime"] = np.vstack(dual_ent_regime_rows)
    return out
