"""
GIC 2026 — Classical Baselines Pipeline
Run: .venv/bin/python main.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

np.random.seed(42)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _ts(label: str) -> None:
    print(f"\n{'='*60}\n{label}\n{'='*60}")


def main():
    from utils.data_loader import load_all
    from utils.features import prepare_all
    from utils.metrics import compute_all

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    _ts("1 / 6  Loading datasets")
    t0 = time.time()
    data = load_all()
    print(f"  → done in {time.time()-t0:.1f}s")
    print(f"  Oxford-Man source: {data['oxford_source']}")

    # ------------------------------------------------------------------ #
    # 2. Feature engineering & splits
    # ------------------------------------------------------------------ #
    _ts("2 / 6  Feature engineering")
    prepared = prepare_all(data)
    fin = prepared["financial"]
    fin_split = prepared["financial_split"]
    vix_split = prepared["vix_split"]
    mg_split = prepared["mackey_glass_split"]
    lor_split = prepared["lorenz_split"]

    n_fin = len(fin)
    n_cut = fin_split["cut"]
    print(f"  Financial rows  : {n_fin}  (train={n_cut}, test={n_fin - n_cut})")
    print(f"  VIX rows        : {len(prepared['vix_df'])}  "
          f"(train={vix_split['cut']}, test={len(prepared['vix_df'])-vix_split['cut']})")
    print(f"  Mackey-Glass    : train={len(mg_split['X_train'])}, test={len(mg_split['X_test'])}")
    print(f"  Lorenz          : train={len(lor_split['X_train'])}, test={len(lor_split['X_test'])}")

    test_dates_fin = fin_split["test"].index
    test_dates_vix = vix_split["test"].index
    y_true_sp500 = fin_split["test"]["target_rv"].values
    y_true_vix_level = vix_split["test"]["target_vix"].values
    y_true_vix_variance = vix_split["test"]["vix_log_ret"].values ** 2  # GARCH target

    # ------------------------------------------------------------------ #
    # 3. Run models
    # ------------------------------------------------------------------ #
    _ts("3 / 6  Running models")

    records = []
    fin_preds = {}
    vix_preds = {}
    mg_preds = {}
    lor_preds = {}
    lstm_losses = {}
    xgb_models = {}

    # ------------------------------------------------------------------ #
    # SP500 RV  (GARCH, HAR-RV, ESN, LSTM, XGBoost)
    # ------------------------------------------------------------------ #
    print("\n[SP500 RV — GARCH]")
    from models.garch import run_garch_on_financial
    garch_sp500 = run_garch_on_financial(fin_split)
    fin_preds["GARCH"] = garch_sp500
    m = compute_all(y_true_sp500, garch_sp500, include_qlike=True)
    records.append({"dataset": "sp500_rv", "model": "GARCH", **m})
    print(f"  RMSE={m['rmse']:.6f}  QLIKE={m['qlike']:.4f}")

    print("\n[SP500 RV — HAR-RV]")
    from models.har_rv import run_har_rv
    har_sp500, har_coefs = run_har_rv(fin_split)
    fin_preds["HAR-RV"] = har_sp500
    m = compute_all(y_true_sp500, har_sp500, include_qlike=True)
    records.append({"dataset": "sp500_rv", "model": "HAR-RV", **m})
    print(f"  RMSE={m['rmse']:.6f}  QLIKE={m['qlike']:.4f}")
    print(f"  coefs: {har_coefs}  sum={sum(v for k,v in har_coefs.items() if k!='intercept'):.4f}")

    print("\n[SP500 RV — ESN]")
    from models.esn import run_esn_financial, run_esn_vix, run_esn_chaotic
    esn_sp500 = run_esn_financial(fin_split)
    fin_preds["ESN"] = esn_sp500
    m = compute_all(y_true_sp500, esn_sp500, include_qlike=True)
    records.append({"dataset": "sp500_rv", "model": "ESN", **m})
    print(f"  RMSE={m['rmse']:.6f}  QLIKE={m['qlike']:.4f}")

    print("\n[SP500 RV — LSTM]")
    from models.lstm import run_lstm_financial, run_lstm_vix, run_lstm_chaotic
    lstm_sp500, loss_sp500 = run_lstm_financial(fin_split)
    fin_preds["LSTM"] = lstm_sp500
    seq_skip_fin = 21  # seq_len - 1: LSTM input ends at X[t], target is y[t] = RV_{t+1}
    m = compute_all(y_true_sp500[seq_skip_fin:], lstm_sp500, include_qlike=True)
    records.append({"dataset": "sp500_rv", "model": "LSTM", **m})
    lstm_losses["sp500_rv"] = loss_sp500
    print(f"  RMSE={m['rmse']:.6f}  QLIKE={m['qlike']:.4f}")

    print("\n[SP500 RV — XGBoost]")
    from models.xgboost_model import run_xgboost_financial, run_xgboost_vix, run_xgboost_chaotic
    xgb_sp500, xgb_sp500_model = run_xgboost_financial(fin_split)
    fin_preds["XGBoost"] = xgb_sp500
    m = compute_all(y_true_sp500, xgb_sp500, include_qlike=True)
    records.append({"dataset": "sp500_rv", "model": "XGBoost", **m})
    xgb_models["sp500_rv"] = xgb_sp500_model
    print(f"  RMSE={m['rmse']:.6f}  QLIKE={m['qlike']:.4f}")

    print("\n[SP500 RV — ARIMA(5,0,0)]")
    from models.arima import run_arima_financial, run_arima_vix
    arima_sp500 = run_arima_financial(fin_split)
    fin_preds["ARIMA"] = arima_sp500
    m = compute_all(y_true_sp500, arima_sp500, include_qlike=True)
    records.append({"dataset": "sp500_rv", "model": "ARIMA", **m})
    print(f"  RMSE={m['rmse']:.6f}  QLIKE={m['qlike']:.4f}")

    # ------------------------------------------------------------------ #
    # Oxford-Man RV  (same data as SP500 RV since GK fallback; separate rows per plan)
    # ------------------------------------------------------------------ #
    print(f"\n[Oxford-Man RV — reusing SP500 predictions (source={data['oxford_source']})]")
    for model_name, pred in [
        ("GARCH", garch_sp500),
        ("HAR-RV", har_sp500),
        ("ESN", esn_sp500),
        ("XGBoost", xgb_sp500),
        ("ARIMA", arima_sp500),
    ]:
        y_t = y_true_sp500
        m = compute_all(y_t, pred, include_qlike=True)
        records.append({"dataset": "oxford_man_rv", "model": model_name, **m})
    # LSTM: aligned length
    m = compute_all(y_true_sp500[seq_skip_fin:], lstm_sp500, include_qlike=True)
    records.append({"dataset": "oxford_man_rv", "model": "LSTM", **m})
    print("  oxford_man_rv rows added (identical to sp500_rv — GK fallback used)")

    # ------------------------------------------------------------------ #
    # VIX  (GARCH on returns/variance, ESN/LSTM/XGBoost on level)
    # ------------------------------------------------------------------ #
    print("\n[VIX — GARCH (conditional variance of VIX log-returns)]")
    from models.garch import run_garch_on_vix
    garch_vix = run_garch_on_vix(vix_split)
    m = compute_all(y_true_vix_variance, garch_vix, include_qlike=True)
    records.append({"dataset": "vix", "model": "GARCH", **m})
    print(f"  target=squared_vix_return  RMSE={m['rmse']:.6f}  QLIKE={m['qlike']:.4f}")

    print("\n[VIX — ESN (level forecast)]")
    esn_vix = run_esn_vix(vix_split)
    vix_preds["ESN"] = esn_vix
    m = compute_all(y_true_vix_level, esn_vix, include_qlike=True)
    records.append({"dataset": "vix", "model": "ESN", **m})
    print(f"  target=vix_level  RMSE={m['rmse']:.4f}  QLIKE={m['qlike']:.4f}")

    print("\n[VIX — LSTM (level forecast)]")
    lstm_vix, loss_vix = run_lstm_vix(vix_split)
    vix_preds["LSTM"] = lstm_vix
    seq_skip_vix = 21  # seq_len - 1
    m = compute_all(y_true_vix_level[seq_skip_vix:], lstm_vix, include_qlike=True)
    records.append({"dataset": "vix", "model": "LSTM", **m})
    lstm_losses["vix"] = loss_vix
    print(f"  target=vix_level  RMSE={m['rmse']:.4f}  QLIKE={m['qlike']:.4f}")

    print("\n[VIX — XGBoost (level forecast)]")
    xgb_vix, xgb_vix_model = run_xgboost_vix(vix_split)
    vix_preds["XGBoost"] = xgb_vix
    m = compute_all(y_true_vix_level, xgb_vix, include_qlike=True)
    records.append({"dataset": "vix", "model": "XGBoost", **m})
    xgb_models["vix"] = xgb_vix_model
    print(f"  target=vix_level  RMSE={m['rmse']:.4f}  QLIKE={m['qlike']:.4f}")

    print("\n[VIX — ARIMA(1,1,0) (level forecast)]")
    arima_vix = run_arima_vix(vix_split)
    vix_preds["ARIMA"] = arima_vix
    m = compute_all(y_true_vix_level, arima_vix, include_qlike=True)
    records.append({"dataset": "vix", "model": "ARIMA", **m})
    print(f"  target=vix_level  RMSE={m['rmse']:.4f}  QLIKE={m['qlike']:.4f}")

    # ------------------------------------------------------------------ #
    # Mackey-Glass  (ESN, LSTM, XGBoost)
    # ------------------------------------------------------------------ #
    print("\n[Mackey-Glass — ESN]")
    esn_mg = run_esn_chaotic(mg_split)
    mg_preds["ESN"] = esn_mg
    m = compute_all(mg_split["y_test"], esn_mg, include_qlike=False)
    records.append({"dataset": "mackey_glass", "model": "ESN", **m})
    print(f"  RMSE={m['rmse']:.6f}")

    print("\n[Mackey-Glass — LSTM]")
    lstm_mg, loss_mg = run_lstm_chaotic(mg_split)
    mg_preds["LSTM"] = lstm_mg
    seq_skip_c = 9  # seq_len - 1
    m = compute_all(mg_split["y_test"][seq_skip_c:], lstm_mg, include_qlike=False)
    records.append({"dataset": "mackey_glass", "model": "LSTM", **m})
    lstm_losses["mackey_glass"] = loss_mg
    print(f"  RMSE={m['rmse']:.6f}")

    print("\n[Mackey-Glass — XGBoost]")
    xgb_mg, xgb_mg_model = run_xgboost_chaotic(mg_split)
    mg_preds["XGBoost"] = xgb_mg
    m = compute_all(mg_split["y_test"], xgb_mg, include_qlike=False)
    records.append({"dataset": "mackey_glass", "model": "XGBoost", **m})
    xgb_models["mackey_glass"] = xgb_mg_model
    print(f"  RMSE={m['rmse']:.6f}")

    # ------------------------------------------------------------------ #
    # Lorenz  (ESN, LSTM, XGBoost)
    # ------------------------------------------------------------------ #
    print("\n[Lorenz — ESN]")
    esn_lor = run_esn_chaotic(lor_split)
    lor_preds["ESN"] = esn_lor
    m = compute_all(lor_split["y_test"], esn_lor, include_qlike=False)
    records.append({"dataset": "lorenz", "model": "ESN", **m})
    print(f"  RMSE={m['rmse']:.6f}")

    print("\n[Lorenz — LSTM]")
    lstm_lor, loss_lor = run_lstm_chaotic(lor_split)
    lor_preds["LSTM"] = lstm_lor
    m = compute_all(lor_split["y_test"][seq_skip_c:], lstm_lor, include_qlike=False)  # seq_skip_c = 9
    records.append({"dataset": "lorenz", "model": "LSTM", **m})
    lstm_losses["lorenz"] = loss_lor
    print(f"  RMSE={m['rmse']:.6f}")

    print("\n[Lorenz — XGBoost]")
    xgb_lor, xgb_lor_model = run_xgboost_chaotic(lor_split)
    lor_preds["XGBoost"] = xgb_lor
    m = compute_all(lor_split["y_test"], xgb_lor, include_qlike=False)
    records.append({"dataset": "lorenz", "model": "XGBoost", **m})
    xgb_models["lorenz"] = xgb_lor_model
    print(f"  RMSE={m['rmse']:.6f}")

    # ------------------------------------------------------------------ #
    # 3b. HMM regime detection + regime accuracy + Sharpe
    # ------------------------------------------------------------------ #
    _ts("3b / 6  HMM regimes + regime accuracy + Sharpe")
    from models.hmm_regimes import fit_hmm, predict_regimes, classify_by_threshold
    from utils.metrics import regime_accuracy, volatility_timing_sharpe

    hmm_model, state_map, hmm_thresholds = fit_hmm(fin_split)
    print(f"  HMM thresholds (RV): low/med={hmm_thresholds[0]:.2e}  med/high={hmm_thresholds[1]:.2e}")

    # True regimes for test period (using actual RV)
    true_regimes_test = predict_regimes(hmm_model, state_map, y_true_sp500)
    regime_counts = {r: int((true_regimes_test == r).sum()) for r in range(3)}
    print(f"  Test regime counts: Low={regime_counts[0]}, Med={regime_counts[1]}, High={regime_counts[2]}")

    # Actual S&P 500 log-returns for test period (for Sharpe calculation)
    test_log_returns = fin_split["test"]["log_ret"].values

    # Buy & hold Sharpe — uses returns[1:] to match the strategy's timing convention
    bnh_returns = test_log_returns[1:]
    bnh_sharpe = float(bnh_returns.mean() / bnh_returns.std() * np.sqrt(252))

    model_regime_acc = {}
    model_sharpes = {}
    for name, pred in fin_preds.items():
        # Align: use last min(len(pred), len(y_true_sp500)) values
        n = min(len(pred), len(y_true_sp500))
        acc = regime_accuracy(true_regimes_test[-n:], pred[-n:], hmm_thresholds)
        sr = volatility_timing_sharpe(pred[-n:], test_log_returns[-n:], hmm_thresholds)
        model_regime_acc[name] = acc
        model_sharpes[name] = sr
        print(f"  {name:10s}  regime_acc={acc:.1%}  Sharpe={sr:.3f}")

    model_sharpes["Buy & Hold"] = bnh_sharpe
    print(f"  {'Buy&Hold':10s}                   Sharpe={bnh_sharpe:.3f}")

    # ------------------------------------------------------------------ #
    # 4. Save metrics
    # ------------------------------------------------------------------ #
    _ts("4 / 6  Saving results")
    metrics_df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"  {len(metrics_df)} rows → {csv_path}")

    # ------------------------------------------------------------------ #
    # 5. Generate plots
    # ------------------------------------------------------------------ #
    _ts("5 / 6  Generating plots")
    from utils.plots import (
        plot_chaotic_forecasts,
        plot_financial_forecasts,
        plot_lstm_losses,
        plot_lorenz_attractor,
        plot_metrics_heatmap,
        plot_radar_chart,
        plot_regime_accuracy,
        plot_regime_dashboard,
        plot_regime_overlay,
        plot_rmse_bars,
        plot_sharpe_comparison,
        plot_xgboost_importance,
    )

    fin_feature_cols = ["rv_d", "rv_w", "rv_m", "log_ret", "vix", "gk", "vix_rv_spread"]
    chaotic_feature_names = [f"lag_{i}" for i in range(10)]
    vix_feature_names = ["vix", "vix_w", "vix_m", "vix_log_ret"]

    # Align fin_preds lengths across models (LSTM has seq_skip fewer points)
    min_len_fin = min(len(v) for v in fin_preds.values())
    fin_preds_aligned = {k: v[-min_len_fin:] for k, v in fin_preds.items()}
    y_true_fin_aligned = y_true_sp500[-min_len_fin:]
    test_dates_fin_aligned = test_dates_fin[-min_len_fin:]

    # Align VIX preds
    min_len_vix = min(len(v) for v in vix_preds.values())
    vix_preds_aligned = {k: v[-min_len_vix:] for k, v in vix_preds.items()}
    y_true_vix_aligned = y_true_vix_level[-min_len_vix:]
    test_dates_vix_aligned = test_dates_vix[-min_len_vix:]

    plot_financial_forecasts(test_dates_fin_aligned, y_true_fin_aligned, fin_preds_aligned, "SP500 RV")
    plot_financial_forecasts(
        test_dates_vix_aligned, y_true_vix_aligned, vix_preds_aligned, "VIX Level",
        ylabel="VIX Level (Index Points)",
    )

    # Align chaotic predictions: LSTM has seq_skip_c=9 fewer points; trim ESN/XGBoost to match
    min_len_mg = min(len(v) for v in mg_preds.values())
    mg_preds_aligned = {k: v[-min_len_mg:] for k, v in mg_preds.items()}
    y_true_mg_aligned = mg_split["y_test"][-min_len_mg:]

    min_len_lor = min(len(v) for v in lor_preds.values())
    lor_preds_aligned = {k: v[-min_len_lor:] for k, v in lor_preds.items()}
    y_true_lor_aligned = lor_split["y_test"][-min_len_lor:]

    plot_chaotic_forecasts(y_true_mg_aligned, mg_preds_aligned, "Mackey-Glass")
    plot_chaotic_forecasts(y_true_lor_aligned, lor_preds_aligned, "Lorenz")

    # Filter out oxford_man_rv and vix-GARCH rows for cleaner heatmap (they're separate targets)
    heatmap_df = metrics_df[metrics_df["dataset"] != "oxford_man_rv"]
    plot_metrics_heatmap(heatmap_df)
    plot_rmse_bars(metrics_df[metrics_df["dataset"].isin(["sp500_rv", "mackey_glass", "lorenz"])])
    plot_radar_chart(metrics_df)
    plot_regime_overlay(
        test_dates_fin_aligned,
        y_true_fin_aligned,
        true_regimes_test[-min_len_fin:],
        fin_preds_aligned,
    )
    plot_regime_dashboard(
        test_dates_fin_aligned,
        y_true_fin_aligned,
        true_regimes_test[-min_len_fin:],
        fin_preds_aligned,
    )
    plot_regime_accuracy(model_regime_acc)
    plot_sharpe_comparison(model_sharpes)
    plot_lstm_losses(lstm_losses)
    plot_lorenz_attractor(data["lorenz"])
    plot_xgboost_importance(
        xgb_models,
        {
            "sp500_rv": fin_feature_cols,
            "vix": vix_feature_names,
            "mackey_glass": chaotic_feature_names,
            "lorenz": chaotic_feature_names,
        },
    )

    # ------------------------------------------------------------------ #
    # 6. Summary
    # ------------------------------------------------------------------ #
    _ts("6 / 6  Summary")
    print(metrics_df.to_string(index=False))
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"\nResults saved to:  {RESULTS_DIR}")
    print(f"Plots saved to:    {RESULTS_DIR / 'plots'}")
    print(f"Oxford-Man source: {data['oxford_source']}")
    print("\nNotes:")
    print("  oxford_man_rv rows = sp500_rv results (GK fallback used — same underlying data)")
    print("  vix GARCH target   = (VIX log-return)² (conditional variance)")
    print("  vix ESN/LSTM/XGB/ARIMA = VIX_{t+1} level")
    print("  ESN/LSTM QLIKE on financial may be unreliable (no positive-output constraint)")
    print("  ARIMA(5,0,0) on RV; ARIMA(1,1,0) on VIX level — finance only")
    print("  HMM regime thresholds (RV): "
          f"low/med={hmm_thresholds[0]:.2e}  med/high={hmm_thresholds[1]:.2e}")


if __name__ == "__main__":
    main()
