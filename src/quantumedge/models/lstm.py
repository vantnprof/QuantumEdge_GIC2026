"""LSTM forecaster using PyTorch."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep
        return self.fc(out).squeeze(-1)


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Build (input_sequence, target) pairs for 1-step-ahead prediction.

    Input window covers steps [t-seq_len+1, t]; target is y[t] (one ahead from X[t]).
    """
    Xs, ys = [], []
    for t in range(seq_len - 1, len(X)):
        Xs.append(X[t - seq_len + 1 : t + 1])  # includes X[t] — most recent features
        ys.append(y[t])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def run_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int = 22,
    hidden: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, list]:
    """Train LSTM; return (test_predictions, train_loss_history)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalise (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Build sequences
    X_tr_seq, y_tr_seq = _make_sequences(X_train_s, y_train, seq_len)
    X_te_seq, _ = _make_sequences(X_test_s, y_test, seq_len)

    # Val split from tail of training
    n_val = max(1, int(len(X_tr_seq) * val_frac))
    X_val_t = torch.tensor(X_tr_seq[-n_val:]).to(device)
    y_val_t = torch.tensor(y_tr_seq[-n_val:]).to(device)
    X_tr_t = torch.tensor(X_tr_seq[:-n_val]).to(device)
    y_tr_t = torch.tensor(y_tr_seq[:-n_val]).to(device)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=False
    )

    model = _LSTMNet(n_features=X_tr_seq.shape[2], hidden=hidden, dropout=dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val, best_state, no_improve = float("inf"), None, 0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr_t)
        loss_history.append(ep_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [LSTM] early stop @ epoch {epoch+1}, best val={best_val:.6f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te_seq).to(device)
        y_pred = model(X_te_t).cpu().numpy()

    return y_pred, loss_history


def run_lstm_financial_log(financial_split: dict, **kwargs) -> tuple[np.ndarray, list]:
    """Log-space LSTM: trains on log(RV) features and log(target_rv), outputs exp(pred).
    Ensures positive predictions and valid QLIKE."""
    raw_cols = ["rv_d", "rv_w", "rv_m", "log_ret", "vix", "gk", "vix_rv_spread"]
    train = financial_split["train"]
    test  = financial_split["test"]

    def _log_features(df):
        X = df[raw_cols].values.copy()
        X[:, :3] = np.log(np.clip(X[:, :3], 1e-12, None))  # log rv_d, rv_w, rv_m
        return X

    y_tr_log = np.log(np.clip(train["target_rv"].values, 1e-12, None))
    y_te_log = np.log(np.clip(test["target_rv"].values,  1e-12, None))

    seq_len = kwargs.pop("seq_len", 22)
    y_pred_log, losses = run_lstm(
        _log_features(train), y_tr_log,
        _log_features(test),  y_te_log,
        seq_len=seq_len, **kwargs,
    )
    return np.exp(y_pred_log), losses


def run_lstm_financial(financial_split: dict, **kwargs) -> tuple[np.ndarray, list]:
    feature_cols = ["rv_d", "rv_w", "rv_m", "log_ret", "vix", "gk", "vix_rv_spread"]
    train = financial_split["train"]
    test = financial_split["test"]
    seq_len = kwargs.pop("seq_len", 22)
    return run_lstm(
        train[feature_cols].values,
        train["target_rv"].values,
        test[feature_cols].values,
        test["target_rv"].values,
        seq_len=seq_len,
        **kwargs,
    )


def run_lstm_vix(vix_split: dict, **kwargs) -> tuple[np.ndarray, list]:
    feature_cols = ["vix", "vix_w", "vix_m", "vix_log_ret"]
    train = vix_split["train"]
    test = vix_split["test"]
    seq_len = kwargs.pop("seq_len", 22)
    return run_lstm(
        train[feature_cols].values,
        train["target_vix"].values,
        test[feature_cols].values,
        test["target_vix"].values,
        seq_len=seq_len,
        **kwargs,
    )


def run_lstm_chaotic(chaotic_split: dict, **kwargs) -> tuple[np.ndarray, list]:
    seq_len = kwargs.pop("seq_len", 10)
    return run_lstm(
        chaotic_split["X_train"],
        chaotic_split["y_train"],
        chaotic_split["X_test"],
        chaotic_split["y_test"],
        seq_len=seq_len,
        **kwargs,
    )
