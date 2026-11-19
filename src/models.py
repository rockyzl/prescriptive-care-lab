"""Predictive models and evaluation metrics (AUROC, AUPRC, Brier, calibration) with bootstrap CIs."""
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def fit_logistic(x, y, p):
    m = LogisticRegression(C=p["C"], penalty=p["penalty"], max_iter=p["max_iter"])
    return m.fit(x, y)


def fit_lgbm(x, y, x_val, y_val, p, seed):
    params = {k: v for k, v in p.items() if k not in ("n_estimators", "early_stopping_rounds")}
    # computational settings for reproducibility (not model hyperparameters)
    params.update(seed=seed, deterministic=True, force_row_wise=True, num_threads=4)
    ds = lgb.Dataset(x, y, free_raw_data=False)
    dv = lgb.Dataset(x_val, y_val, reference=ds)
    booster = lgb.train(params, ds, num_boost_round=p["n_estimators"], valid_sets=[dv], valid_names=["val"],
                        callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False)])
    return booster


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _newton_logistic(X, y, offset=None, iters=50):
    """Unpenalized logistic regression by Newton-Raphson (small number of columns)."""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def calibration_intercept_slope(y, p):
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope."""
    lp = logit(p)
    y = np.asarray(y, float)
    a = _newton_logistic(np.ones((len(y), 1)), y, offset=lp)[0]
    _, b = _newton_logistic(np.column_stack([np.ones(len(y)), lp]), y)
    return float(a), float(b)


def reliability(y, p, bins=10):
    """Quantile-binned reliability curve: mean predicted vs observed rate per bin."""
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    return [{"mean_pred": float(p[g].mean()), "obs_rate": float(np.mean(y[g])), "n": int(len(g))} for g in groups]


def point_metrics(y, p):
    y = np.asarray(y)
    a, b = calibration_intercept_slope(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "cal_intercept": a,
        "cal_slope": b,
        "mean_pred": float(np.mean(p)),
        "obs_rate": float(np.mean(y)),
        "n": int(len(y)),
        "events": int(np.sum(y)),
    }


def ci(values):
    v = np.asarray(values, float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bootstrap_metrics(y, preds, n_boot, seed, pairs=()):
    """Beneficiary bootstrap. preds: name -> prob array. Same resamples for every model (paired)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    keys = ["auroc", "auprc", "brier", "cal_intercept", "cal_slope"]
    draws = {m: {k: [] for k in keys} for m in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for m, p in preds.items():
            pm = point_metrics(yb, p[idx])
            for k in keys:
                draws[m][k].append(pm[k])
    out = {}
    for m, p in preds.items():
        pm = point_metrics(y, p)
        out[m] = {k: {"estimate": pm[k], "ci95": ci(draws[m][k])} for k in keys}
        out[m].update(n=pm["n"], events=pm["events"], mean_pred=pm["mean_pred"], obs_rate=pm["obs_rate"])
    diffs = {}
    for a, b in pairs:
        diffs[f"{a}-{b}"] = {k: {"estimate": out[a][k]["estimate"] - out[b][k]["estimate"],
                                 "ci95": ci(np.array(draws[a][k]) - np.array(draws[b][k]))} for k in ["auroc", "auprc", "brier"]}
    return out, diffs
