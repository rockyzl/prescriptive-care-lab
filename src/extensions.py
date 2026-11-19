"""Addendum modules (protocol Addendum A and B), run after the frozen one-shot test.

  python -m src.extensions causal     -> results/causal_results.json
  python -m src.extensions forecast   -> results/forecast_results.json
"""
import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone

import numpy as np

from src.causal import balance, estimate, simulate, standardize
from src.data import ROOT
from src.models import ci, logit
from src.run import RESULTS, load, load_artefacts, peak_rss_mb, predict_all, write_json

SEED = 2026
N_REPS = 100
N_BOOT = 200
ESTIMATORS = ["naive", "gcomp", "ipw", "aipw"]
SCENARIOS = [{"gamma": g, "rrr": r, "hidden": False} for g in (0.0, 0.5, 1.0, 2.0) for r in (0.0, 0.15, 0.30)] + \
            [{"gamma": 1.0, "rrr": r, "hidden": True} for r in (0.0, 0.15)]
_G = {}


def _init(x, y0, s):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _G.update(x=x, y0=y0, s=s)


def _sim(si, rep):
    sc = SCENARIOS[si]
    rng = np.random.default_rng([SEED, si, rep])
    return simulate(_G["y0"], _G["s"], sc["gamma"], sc["rrr"], rng, hidden_weight=sc["hidden"])


def _rep_task(args):
    si, rep = args
    t, y, _, true_ate = _sim(si, rep)
    est = estimate(_G["x"], t, y)
    est.pop("_e")
    return si, rep, true_ate, est


def _boot_task(args):
    si, b = args
    t, y, _, _ = _sim(si, 0)
    idx = np.random.default_rng([SEED, si, 10_000 + b]).integers(0, len(y), len(y))
    est = estimate(_G["x"][idx], t[idx], y[idx])
    return si, {k: est[k] for k in ESTIMATORS}


def causal():
    t0 = time.time()
    feats, _, labels = load(["test"])
    lab = labels["test"]
    te = feats.loc[lab.index]
    art = load_artefacts()
    from src.features import Preprocessor
    x = Preprocessor.from_dict(art["preprocessor"]).transform(te, True)
    risk = predict_all(te, art)[("y_ip", "lightgbm")]
    s = standardize(logit(risk))
    y0 = lab.y_ip.to_numpy().astype(int)

    ctx = mp.get_context("fork")
    with ctx.Pool(8, initializer=_init, initargs=(x, y0, s)) as pool:
        reps = pool.map(_rep_task, [(si, r) for si in range(len(SCENARIOS)) for r in range(N_REPS)], chunksize=10)
        boots = pool.map(_boot_task, [(si, b) for si in range(len(SCENARIOS)) for b in range(N_BOOT)], chunksize=10)
    _init(x, y0, s)
    out = []
    cols = art["preprocessor"]["columns"]
    for si, sc in enumerate(SCENARIOS):
        rr = sorted([r for r in reps if r[0] == si], key=lambda r: r[1])
        t, y, _, true1 = _sim(si, 0)
        est1 = estimate(x, t, y)
        bal = balance(x, t, est1.pop("_e"))
        keep_cov = sc["gamma"] == 2.0 and sc["rrr"] == 0.15 and not sc["hidden"]
        if keep_cov:
            bal["covariates"] = cols
        else:
            bal.pop("smd_before"), bal.pop("smd_after_ipw")
        bdraws = {k: [b[1][k] for b in boots if b[0] == si] for k in ESTIMATORS}
        across = {}
        for k in ESTIMATORS:
            bias = np.array([r[3][k] - r[2] for r in rr])
            d = {"mean_bias": float(bias.mean()), "rmse": float(np.sqrt(np.mean(bias ** 2))), "bias_mcse": float(bias.std(ddof=1) / np.sqrt(len(bias)))}
            if k in ("ipw", "aipw"):
                cover = [abs(r[3][k] - r[2]) <= 1.96 * r[3][k + "_se"] for r in rr]
                d["wald_coverage95"] = float(np.mean(cover))
            d["recovers"] = bool(abs(d["mean_bias"]) < 0.005 and d.get("wald_coverage95", 1.0) >= 0.90)
            across[k] = d
        out.append({**sc, "enrollment_target": 0.2,
                    "replicate1": {"true_ate": true1, "estimates": {k: {"estimate": est1[k], "bootstrap_ci95": ci(bdraws[k])} for k in ESTIMATORS},
                                   "ipw_if_se": est1["ipw_se"], "aipw_if_se": est1["aipw_se"], "balance": bal},
                    "across_replicates": {"n_reps": len(rr), "mean_true_ate": float(np.mean([r[2] for r in rr])), "estimators": across}})
        print(sc, "true", round(true1, 4), {k: round(est1[k], 4) for k in ESTIMATORS}, {k: round(v["mean_bias"], 4) for k, v in across.items()}, flush=True)
    res = {"module": "Addendum A: semi-synthetic program-effect estimation", "created_at": datetime.now(timezone.utc).isoformat(),
           "warning": "Enrollment and program effect are SIMULATED on observed 2009 outcomes of synthetic (DE-SynPUF) test members. This estimates no real program effect.",
           "n": int(len(y0)), "baseline_admission_rate": float(y0.mean()), "n_reps": N_REPS, "n_boot": N_BOOT,
           "estimand": "sample ATE on the risk-difference scale, mean(Y1 - Y0)", "scenarios": out,
           "compute": {"seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "workers": 8}}
    write_json(RESULTS / "causal_results.json", res)
    print("causal done", round(time.time() - t0), "s")


def forecast():
    from src.forecast import WINDOW, backtest, in_window, monthly_series, summarize
    t0 = time.time()
    full = monthly_series()
    win = full[in_window(full.index)]
    methods = ["naive", "seasonal_naive", "drift", "ets_damped"]
    out = {"module": "Addendum B: monthly admission forecasting", "created_at": datetime.now(timezone.utc).isoformat(),
           "warning": "DE-SynPUF is synthetic; the series is a closed 2008 cohort with documented 2010 attrition/disclosure artefacts.",
           "full_series": {str(k): {"admissions": int(v.admissions), "payments": float(v.payments)} for k, v in full.iterrows()},
           "window": list(WINDOW), "n_months_window": int(len(win)), "methods": methods, "series": {}}
    for name in ["admissions", "payments"]:
        y = win[name].to_numpy(float)
        bt = backtest(y, methods)
        scale = float(np.mean(np.abs(np.diff(y[:12]))))
        bt["target_month"] = [int(win.index[i]) for i in bt.target_pos]
        out["series"][name] = {"mase_scale_first_window_naive_mae": scale, "summary": summarize(bt, scale),
                               "backtest": bt.drop(columns=["target_pos"]).to_dict(orient="records")}
        print(name, {k: (round(v["mase"], 2), v["coverage80"], v["coverage95"]) for k, v in out["series"][name]["summary"].items()}, flush=True)
    out["compute"] = {"seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb()}
    write_json(RESULTS / "forecast_results.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("module", choices=["causal", "forecast"])
    {"causal": causal, "forecast": forecast}[ap.parse_args().module]()
