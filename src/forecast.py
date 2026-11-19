"""Monthly inpatient-admission forecasting with a rolling-origin backtest (protocol Addendum B)."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from src.data import read_zip_csv

WINDOW = (200803, 200912)
Z = {0.8: norm.ppf(0.9), 0.95: norm.ppf(0.975)}


def monthly_series():
    ip = read_zip_csv("inpatient", usecols=["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_ADMSN_DT", "CLM_PMT_AMT"], dtype={"CLM_ID": str})
    return series_from_claims(ip)


def series_from_claims(ip):
    ip = ip.drop_duplicates(["DESYNPUF_ID", "CLM_ID"])
    d = pd.to_numeric(ip.CLM_ADMSN_DT, errors="coerce").fillna(pd.to_numeric(ip.CLM_FROM_DT, errors="coerce"))
    ym = (d // 100).astype(int)
    return pd.DataFrame({"admissions": ip.groupby(ym).size(), "payments": ip.CLM_PMT_AMT.groupby(ym).sum()}).sort_index()


def in_window(index, window=WINDOW):
    idx = np.asarray(index)
    return (idx >= window[0]) & (idx <= window[1])


def _sd(resid):
    resid = np.asarray(resid, float)
    return float(np.std(resid, ddof=1)) if len(resid) >= 2 else np.nan


def forecast(train, h, method, seed=2026):
    """Point forecasts and interval bounds for horizons 1..h using ONLY `train`."""
    y = np.asarray(train, float)
    n = len(y)
    hs = np.arange(1, h + 1)
    naive_sd = _sd(np.diff(y))
    if method == "naive":
        point, sd = np.repeat(y[-1], h), naive_sd * np.sqrt(hs)
    elif method == "seasonal_naive":
        point = np.array([y[n - 1 + k - 12] for k in hs])
        s = _sd(y[12:] - y[:-12]) if n >= 14 else np.nan
        sd = (s if np.isfinite(s) else naive_sd) * np.sqrt(hs)
    elif method == "drift":
        slope = (y[-1] - y[0]) / (n - 1)
        point = y[-1] + slope * hs
        sd = _sd(np.diff(y) - slope) * np.sqrt(hs)
    elif method == "ets_damped":
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        ys = pd.Series(y, index=pd.period_range("2000-01", periods=n, freq="M"))  # dummy monthly index (statsmodels needs one)
        res = ETSModel(ys, error="add", trend="add", damped_trend=True).fit(disp=False, maxiter=2000)
        pred = res.get_prediction(start=n, end=n + h - 1, method="simulated", simulate_repetitions=1000, rng=np.random.default_rng(seed))
        out = {"point": np.asarray(pred.predicted_mean, float)}
        for level in (0.8, 0.95):
            sf = pred.summary_frame(alpha=1 - level)
            out[level] = (sf["pi_lower"].to_numpy(float), sf["pi_upper"].to_numpy(float))
        return out
    else:
        raise ValueError(method)
    out = {"point": point}
    for level, z in Z.items():
        out[level] = (point - z * sd, point + z * sd)
    return out


def backtest(series, methods, first_train=12, horizons=(1, 3)):
    """Expanding-window rolling origin. Each forecast uses series[:origin] only."""
    y = np.asarray(series, float)
    n, hmax = len(y), max(horizons)
    rows = []
    for origin in range(first_train, n):
        train = y[:origin]
        for m in methods:
            f = forecast(train, hmax, m)
            for h in horizons:
                tgt = origin + h - 1
                if tgt >= n:
                    continue
                rows.append({"method": m, "h": h, "origin_len": origin, "target_pos": tgt, "actual": y[tgt], "point": float(f["point"][h - 1]),
                             **{f"lo{int(l*100)}": float(f[l][0][h - 1]) for l in (0.8, 0.95)},
                             **{f"hi{int(l*100)}": float(f[l][1][h - 1]) for l in (0.8, 0.95)}})
    return pd.DataFrame(rows)


def summarize(bt, scale):
    out = {}
    for (m, h), g in bt.groupby(["method", "h"]):
        err = g.actual - g.point
        d = {"n_forecasts": int(len(g)), "mae": float(err.abs().mean()), "mape_pct": float((err.abs() / g.actual).mean() * 100),
             "mase": float(err.abs().mean() / scale), "mean_error": float(err.mean())}
        for l in (80, 95):
            k = int(((g.actual >= g[f"lo{l}"]) & (g.actual <= g[f"hi{l}"])).sum())
            ci = binomtest(k, len(g)).proportion_ci(confidence_level=0.95, method="exact")
            d[f"coverage{l}"] = k / len(g)
            d[f"coverage{l}_exact_ci95"] = [float(ci.low), float(ci.high)]
        out[f"{m}|h={h}"] = d
    return out
