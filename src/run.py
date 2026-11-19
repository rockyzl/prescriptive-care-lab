"""Phases: dev (train + validation only) -> freeze -> test (one shot).

  python -m src.run dev
  python -m src.run freeze
  python -m src.run test
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.cluster import cluster_matrix_fit, linear_by_rank, select_k
from src.data import DERIVED, ROOT, sha256_file
from src.features import Preprocessor, high_cost_threshold, predictor_columns
from src.models import bootstrap_metrics, ci, fit_lgbm, fit_logistic, point_metrics, reliability
from src.prescriptive import design_effect, evaluate, lp_upper_bound, random_policy, select_all, two_proportion_n

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
OUTCOMES = ["y_ip", "high_cost"]
FROZEN_FILES = ["protocol.json", "pyproject.toml", "uv.lock", "src/__init__.py", "src/data.py", "src/features.py", "src/models.py",
                "src/cluster.py", "src/prescriptive.py", "src/run.py", "data/manifest.json", "data/membership.json",
                "results/dev_results.json", "models/preprocessor.json", "models/logistic.json", "models/lgbm_y_ip.txt",
                "models/lgbm_high_cost.txt", "models/phenotypes.json", "models/dev_meta.json"]


def protocol():
    return json.loads((ROOT / "protocol.json").read_text())


def _clean(v):
    """NaN/inf -> None so results stay strict JSON (e.g. SHAP correlation for an unused feature)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def write_json(path, value, exclusive=False):
    value = _clean(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x" if exclusive else "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def env_info():
    import lightgbm, pulp, sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__, "pulp": pulp.__version__}


def load(parts):
    """Features for every cohort member (no outcomes) + labels for the requested parts ONLY."""
    feats = pd.read_parquet(DERIVED / "features.parquet").set_index("DESYNPUF_ID")
    audit = pd.read_parquet(DERIVED / "audit.parquet").set_index("DESYNPUF_ID")
    labels = {}
    for part in parts:
        lab = pd.read_parquet(DERIVED / f"labels_{part}.parquet").set_index("DESYNPUF_ID")
        if not (feats.loc[lab.index, "split"] == part).all():
            raise ValueError(f"labels_{part} contains members from another split")
        labels[part] = lab
    return feats, audit, labels


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_logistic(coef, x):
    return sigmoid(x @ np.asarray(coef["coef"]) + coef["intercept"])


def predict_all(feats_part, art):
    import lightgbm as lgb
    pre = Preprocessor.from_dict(art["preprocessor"])
    xs, xr = pre.transform(feats_part, True), pre.transform(feats_part, False)
    out = {}
    for o in OUTCOMES:
        out[(o, "logistic")] = predict_logistic(art["logistic"][o], xs)
        out[(o, "lightgbm")] = lgb.Booster(model_file=str(MODELS / f"lgbm_{o}.txt")).predict(xr)
    return out


def load_artefacts():
    return {"preprocessor": json.loads((MODELS / "preprocessor.json").read_text()),
            "logistic": json.loads((MODELS / "logistic.json").read_text()),
            "phenotypes": json.loads((MODELS / "phenotypes.json").read_text()),
            "meta": json.loads((MODELS / "dev_meta.json").read_text())}


def assign_phenotype(feats_part, ph):
    pre = Preprocessor.from_dict(ph["preprocessor"])
    x = pre.transform(feats_part, True)
    c = np.asarray(ph["centroids"])
    d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def scenario_vectors(phen, ph, p):
    """Per-member assumed RRR (per effect scenario) and contact cost (per cost scenario)."""
    rank = np.asarray(ph["risk_rank"])[phen]
    k = ph["k"]
    pr = p["prescriptive"]
    effects = {}
    for name, sc in pr["effect_scenarios"].items():
        if sc["type"] == "uniform":
            effects[name] = np.full(len(phen), sc["rrr"])
        else:
            effects[name] = np.array([linear_by_rank(r, k, sc["low_risk_rrr"], sc["high_risk_rrr"]) for r in range(k)])[rank]
    costs = {}
    for name, sc in pr["cost_scenarios"].items():
        if sc["type"] == "count":
            costs[name] = np.ones(len(phen))
        else:
            costs[name] = np.array([linear_by_rank(r, k, sc["low_risk_hours"], sc["high_risk_hours"]) for r in range(k)])[rank]
    return effects, costs


def budget_for(cost_name, frac, n, p):
    slots = int(round(frac * n))
    sc = p["prescriptive"]["cost_scenarios"][cost_name]
    return slots if sc["type"] == "count" else slots * sc["budget_hours_per_capacity_slot"], slots


def random_summary(y, r, pr, cost, budget, draws, rng):
    vals = [evaluate(random_policy(len(y), cost, budget, rng), y, r, pr, cost) for _ in range(draws)]
    out = {}
    for key in ["contacted", "admissions_captured", "realized_avoided", "planned_avoided"]:
        v = np.array([d[key] for d in vals])
        out[key] = float(v.mean())
        out[key + "_p2.5_p97.5"] = ci(v)
    return out


def prescriptive_analysis(y, risk, prior_cost, phen, ph, p, n_boot, seed, use_ilp=True, random_draws=None):
    effects, costs = scenario_vectors(phen, ph, p)
    pr = p["prescriptive"]
    rng = np.random.default_rng(seed)
    draws = random_draws or pr["random_policy_draws"]
    n = len(y)
    results = []
    idx_boot = [rng.integers(0, n, n) for _ in range(n_boot)]
    for cost_name, cost in costs.items():
        for frac in pr["capacity_fractions"]:
            budget, slots = budget_for(cost_name, frac, n, p)
            for eff_name, r in effects.items():
                masks, status = select_all(risk, r, cost, prior_cost, y, budget, cost_name == "uniform", use_ilp=use_ilp)
                point = {k: evaluate(m, y, r, risk, cost) for k, m in masks.items()}
                greedy_masks, _ = select_all(risk, r, cost, prior_cost, y, budget, cost_name == "uniform", use_ilp=False)
                gopt = evaluate(greedy_masks["optimized"], y, r, risk, cost)
                point["random"] = random_summary(y, r, risk, cost, budget, draws, rng)
                # bootstrap with re-selection (greedy == ILP for uniform cost; greedy ratio rule for hours)
                bd = {k: {"realized_avoided": [], "admissions_captured": []} for k in ["highest_cost", "highest_risk", "optimized", "random"]}
                for idx in idx_boot:
                    yb, rb, pb, cb, prb = y[idx], r[idx], risk[idx], cost[idx], prior_cost[idx]
                    mb, _ = select_all(pb, rb, cb, prb, yb, budget, cost_name == "uniform", use_ilp=False)
                    for k in ["highest_cost", "highest_risk", "optimized"]:
                        e = evaluate(mb[k], yb, rb, pb, cb)
                        bd[k]["realized_avoided"].append(e["realized_avoided"])
                        bd[k]["admissions_captured"].append(e["admissions_captured"])
                    if cost_name == "uniform":  # exact expectation of random selection of `slots` members
                        bd["random"]["realized_avoided"].append(slots * float(np.mean(yb * rb)))
                        bd["random"]["admissions_captured"].append(slots * float(np.mean(yb)))
                    else:
                        rr = [evaluate(random_policy(n, cb, budget, rng), yb, rb, pb, cb) for _ in range(20)]
                        bd["random"]["realized_avoided"].append(float(np.mean([e["realized_avoided"] for e in rr])))
                        bd["random"]["admissions_captured"].append(float(np.mean([e["admissions_captured"] for e in rr])))
                cis = {k: {m: ci(v) for m, v in d.items()} for k, d in bd.items()}
                diffs = {}
                for a, b in [("optimized", "highest_risk"), ("optimized", "highest_cost"), ("highest_risk", "highest_cost"),
                             ("highest_risk", "random"), ("optimized", "random")]:
                    diffs[f"{a}-{b}"] = {}
                    for m in ["realized_avoided", "admissions_captured"]:
                        est = point[a][m] - point[b][m]
                        diffs[f"{a}-{b}"][m] = {"estimate": est, "ci95": ci(np.array(bd[a][m]) - np.array(bd[b][m]))}
                results.append({"cost_scenario": cost_name, "capacity_fraction": frac, "capacity_slots": slots, "budget": budget,
                                 "effect_scenario": eff_name, "ilp_status": status, "point": point, "bootstrap_ci95": cis,
                                 "paired_differences": diffs,
                                 "greedy_optimized_realized_avoided": gopt["realized_avoided"],
                                 "greedy_optimized_planned_avoided": gopt["planned_avoided"],
                                 "ilp_minus_greedy_planned_avoided": point["optimized"]["planned_avoided"] - gopt["planned_avoided"],
                                 "lp_upper_bound_planned_avoided": lp_upper_bound(risk * r, cost, budget)})
                print(f"  {cost_name:7s} {frac:.2f} {eff_name:5s} opt={point['optimized']['realized_avoided']:.1f} "
                      f"risk={point['highest_risk']['realized_avoided']:.1f} cost={point['highest_cost']['realized_avoided']:.1f} "
                      f"rand={point['random']['realized_avoided']:.1f} [{status}]", flush=True)
    return results


# ----------------------------------------------------------------------------- dev
def dev():
    t0 = time.time()
    p = protocol()
    seed = p["seed"]
    feats, audit, labels = load(["train", "val"])
    cols = predictor_columns(feats, p)
    tr, va = feats.loc[labels["train"].index], feats.loc[labels["val"].index]
    thr = high_cost_threshold(labels["train"].cost_2009, p["high_cost_quantile"])
    ys = {part: {"y_ip": labels[part].y_ip.to_numpy(), "high_cost": (labels[part].cost_2009 >= thr).astype(int).to_numpy()} for part in labels}

    pre = Preprocessor(cols, p["log1p_features"]).fit(tr)
    xs_tr, xs_va = pre.transform(tr, True), pre.transform(va, True)
    xr_tr, xr_va = pre.transform(tr, False), pre.transform(va, False)
    MODELS.mkdir(exist_ok=True)
    write_json(MODELS / "preprocessor.json", pre.to_dict())

    val_metrics, logistic, lgbm_info = {}, {}, {}
    for o in OUTCOMES:
        lr = fit_logistic(xs_tr, ys["train"][o], p["logistic"])
        logistic[o] = {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "columns": cols}
        booster = fit_lgbm(xr_tr, ys["train"][o], xr_va, ys["val"][o], p["lightgbm"], seed)
        booster.save_model(str(MODELS / f"lgbm_{o}.txt"), num_iteration=booster.best_iteration)
        lgbm_info[o] = {"best_iteration": int(booster.best_iteration)}
        val_metrics[o] = {"logistic": point_metrics(ys["val"][o], lr.predict_proba(xs_va)[:, 1]),
                          "lightgbm": point_metrics(ys["val"][o], booster.predict(xr_va, num_iteration=booster.best_iteration))}
        print(o, {m: round(v["auroc"], 4) for m, v in val_metrics[o].items()}, lgbm_info[o], flush=True)
    write_json(MODELS / "logistic.json", logistic)
    chosen = max(["logistic", "lightgbm"], key=lambda m: val_metrics["y_ip"][m]["auroc"])

    # SHAP (validation subsample, interpretation only)
    import lightgbm as lgb
    import shap
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(va), p["shap"]["rows"], replace=False)
    booster = lgb.Booster(model_file=str(MODELS / "lgbm_y_ip.txt"))
    sv = shap.TreeExplainer(booster).shap_values(xr_va[sub])
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(-mean_abs)
    shap_rank = [{"feature": cols[i], "mean_abs_shap": float(mean_abs[i]),
                  "corr_feature_shap": float(np.corrcoef(xr_va[sub][:, i], sv[:, i])[0, 1]) if np.std(xr_va[sub][:, i]) > 0 else None}
                 for i in order]
    lr_coef = sorted([{"feature": c, "coef_per_sd": float(b)} for c, b in zip(cols, logistic["y_ip"]["coef"])], key=lambda d: -abs(d["coef_per_sd"]))

    # Phenotyping (train only)
    cfg = p["clustering"]
    cpre = cluster_matrix_fit(tr, cfg)
    x_c = cpre.transform(tr, True)
    k_table, (k, rule) = select_k(x_c, cfg, seed)
    from sklearn.cluster import KMeans
    km = KMeans(k, n_init=cfg["kmeans_n_init"], random_state=seed).fit(x_c)
    lab_tr = km.labels_
    adm_rate = pd.Series(ys["train"]["y_ip"]).groupby(lab_tr).mean()
    rank_order = adm_rate.sort_values(kind="stable").index.tolist()
    risk_rank = [rank_order.index(c) for c in range(k)]
    profile_cols = cfg["features"] + ["age", "female", "n_chronic", "ip_paid"]
    profiles = []
    for c in range(k):
        m = lab_tr == c
        prof = tr.loc[m, profile_cols].mean().to_dict()
        profiles.append({"phenotype": c, "risk_rank": risk_rank[c], "n_train": int(m.sum()), "share_train": float(m.mean()),
                         "train_admission_rate_2009": float(adm_rate[c]),
                         "train_high_cost_rate_2009": float(ys["train"]["high_cost"][m].mean()),
                         "profile_means": {kk: float(v) for kk, v in prof.items()}})
    ph = {"k": int(k), "selection_rule": rule, "preprocessor": cpre.to_dict(), "centroids": km.cluster_centers_.tolist(), "risk_rank": risk_rank}
    eff, cst = scenario_vectors(np.arange(k), ph, p)
    ph["assumed_rrr_by_phenotype"] = {n: v.tolist() for n, v in eff.items()}
    ph["assumed_hours_by_phenotype"] = {n: v.tolist() for n, v in cst.items()}
    write_json(MODELS / "phenotypes.json", ph)
    write_json(MODELS / "dev_meta.json", {"high_cost_threshold": thr, "targeting_model": chosen, "predictor_columns": cols, "lightgbm": lgbm_info})

    # Rehearsal of the prescriptive pipeline on VALIDATION (catches bugs before the one-shot test)
    art = load_artefacts()
    preds_va = predict_all(va, art)
    phen_va = assign_phenotype(va, ph)
    rehearsal = prescriptive_analysis(ys["val"]["y_ip"], preds_va[("y_ip", chosen)], va.bene08_medreimb_total.to_numpy(float), phen_va, ph, p,
                                      n_boot=20, seed=seed, use_ilp=False, random_draws=50)

    out = {
        "phase": "dev", "created_at": datetime.now(timezone.utc).isoformat(), "env": env_info(),
        "n": {"train": len(tr), "val": len(va)}, "events_y_ip": {k: int(v["y_ip"].sum()) for k, v in ys.items()},
        "high_cost_threshold_train_q90": thr, "high_cost_prevalence": {k: float(v["high_cost"].mean()) for k, v in ys.items()},
        "n_predictors": len(cols), "predictor_columns": cols,
        "validation_metrics": val_metrics, "lightgbm": lgbm_info, "targeting_model": chosen,
        "targeting_model_rule": "higher validation AUROC for y_ip",
        "shap_val_subsample": shap_rank, "logistic_coefficients_y_ip": lr_coef,
        "clustering": {"k_table": k_table, "chosen_k": int(k), "rule": rule, "profiles_train": profiles,
                       "assumed_rrr_by_phenotype": ph["assumed_rrr_by_phenotype"], "assumed_hours_by_phenotype": ph["assumed_hours_by_phenotype"]},
        "prescriptive_rehearsal_on_validation": {"note": "Pipeline rehearsal on the validation split (20 bootstrap resamples, 50 random draws, greedy instead of ILP for speed); not a result.",
                                                  "results": rehearsal},
        "compute": {"seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb()},
    }
    write_json(RESULTS / "dev_results.json", out)
    print("dev done", round(time.time() - t0), "s", "peak RSS MB", round(peak_rss_mb()))


# ----------------------------------------------------------------------------- freeze / test
def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def freeze():
    dirty = git("status", "--porcelain", "--", *FROZEN_FILES).strip()
    if dirty:
        raise SystemExit(f"Refusing to freeze: uncommitted frozen files:\n{dirty}")
    tracked = set(git("ls-files").split())
    missing = [f for f in FROZEN_FILES if f not in tracked]
    if missing:
        raise SystemExit(f"Refusing to freeze: untracked frozen files {missing}")
    if (RESULTS / "test_results.json").exists():
        raise SystemExit("Refusing to freeze: test already run")
    out = {"commit": git("rev-parse", "HEAD").strip(), "frozen_at": datetime.now(timezone.utc).isoformat(),
           "sha256": {f: sha256_file(ROOT / f) for f in FROZEN_FILES},
           "derived_sha256": {f: sha256_file(DERIVED / f) for f in ["features.parquet", "audit.parquet", "labels_train.parquet",
                                                                     "labels_val.parquet", "labels_test.parquet"]}}
    write_json(RESULTS / "freeze.json", out)
    print("frozen at", out["commit"], "- commit results/freeze.json, then run `python -m src.run test`")


def check_freeze():
    fz = RESULTS / "freeze.json"
    if not fz.exists() or git("status", "--porcelain", "--", "results/freeze.json").strip() or "results/freeze.json" not in git("ls-files"):
        raise SystemExit("Refusing to test: results/freeze.json missing or uncommitted")
    frozen = json.loads(fz.read_text())
    bad = [f for f, h in frozen["sha256"].items() if sha256_file(ROOT / f) != h]
    bad += [f for f, h in frozen["derived_sha256"].items() if sha256_file(DERIVED / f) != h]
    if bad:
        raise SystemExit(f"Refusing to test: changed since freeze: {bad}")
    return frozen


def subgroup_audit(y, risk, sel, audit_part, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx_boot = [rng.integers(0, n, n) for _ in range(n_boot)]
    from sklearn.metrics import roc_auc_score
    overall_capture = y[sel].sum() / y.sum()
    boot_overall = [y[i][sel[i]].sum() / max(y[i].sum(), 1) for i in idx_boot]
    out = {"overall": {"capture_rate": float(overall_capture), "capture_rate_ci95": ci(boot_overall),
                       "selection_rate": float(sel.mean()), "admission_rate": float(y.mean())}, "groups": {}}
    for field in ["age_group", "sex", "race"]:
        g = audit_part[field].to_numpy()
        for level in sorted(pd.unique(g)):
            m = g == level
            cap = y[m & sel].sum() / max(y[m].sum(), 1)
            bcap, bauc = [], []
            for i in idx_boot:
                mi = m[i]
                bcap.append(y[i][mi & sel[i]].sum() / max(y[i][mi].sum(), 1))
                if 0 < y[i][mi].sum() < mi.sum():
                    bauc.append(roc_auc_score(y[i][mi], risk[i][mi]))
            capci = ci(bcap)
            out["groups"][f"{field}={level}"] = {
                "n": int(m.sum()), "events": int(y[m].sum()), "admission_rate": float(y[m].mean()),
                "mean_predicted": float(risk[m].mean()), "calibration_in_the_large": float(risk[m].mean() - y[m].mean()),
                "auroc": float(roc_auc_score(y[m], risk[m])), "auroc_ci95": ci(bauc),
                "selection_rate": float(sel[m].mean()), "capture_rate": float(cap), "capture_rate_ci95": capci,
                "flag_capture_gap_gt_5pp": bool(abs(cap - overall_capture) > 0.05 and not (capci[0] <= overall_capture <= capci[1])),
            }
    return out


def test():
    frozen = check_freeze()
    p = protocol()
    evaluate_part("test", p["bootstrap"]["metrics_resamples"], p["bootstrap"]["policy_resamples"], use_ilp=True,
                  out_path=RESULTS / "test_results.json", exclusive=True, freeze_commit=frozen["commit"])


def dryrun():
    """Rehearse the full evaluation on the VALIDATION split (never reads test labels, never writes test_results.json)."""
    evaluate_part("val", 50, 20, use_ilp=False, out_path=RESULTS / "val_dryrun.json", exclusive=False, freeze_commit=None)


def evaluate_part(part, n_metric_boot, n_policy_boot, use_ilp, out_path, exclusive, freeze_commit):
    if part not in ("val", "test"):
        raise ValueError(part)
    t0 = time.time()
    p = protocol()
    seed = p["seed"]
    feats, audit, labels = load([part])
    lab = labels[part]
    te, au = feats.loc[lab.index], audit.loc[lab.index]
    art = load_artefacts()
    thr = art["meta"]["high_cost_threshold"]
    chosen = art["meta"]["targeting_model"]
    ys = {"y_ip": lab.y_ip.to_numpy(), "high_cost": (lab.cost_2009 >= thr).astype(int).to_numpy()}
    preds = predict_all(te, art)

    predictive = {}
    for o in OUTCOMES:
        pm = {m: preds[(o, m)] for m in ["logistic", "lightgbm"]}
        res, diffs = bootstrap_metrics(ys[o], pm, n_metric_boot, seed, pairs=[("lightgbm", "logistic")])
        predictive[o] = {"metrics": res, "paired_differences": diffs, "prevalence": float(ys[o].mean()),
                         "reliability": {m: reliability(ys[o], v, p["calibration_bins"]) for m, v in pm.items()}}
        print(o, {m: round(v["auroc"]["estimate"], 4) for m, v in res.items()}, flush=True)

    risk = preds[("y_ip", chosen)]
    y = ys["y_ip"]
    ph = art["phenotypes"]
    phen = assign_phenotype(te, ph)
    by_phen = []
    for c in range(ph["k"]):
        m = phen == c
        by_phen.append({"phenotype": c, "risk_rank": ph["risk_rank"][c], "n_test": int(m.sum()), "share_test": float(m.mean()),
                        "test_admission_rate_2009": float(y[m].mean()), "test_high_cost_rate_2009": float(ys["high_cost"][m].mean()),
                        "test_mean_predicted_risk": float(risk[m].mean())})
    print("phenotypes", [(d["phenotype"], round(d["test_admission_rate_2009"], 3)) for d in by_phen], flush=True)

    prior_cost = te.bene08_medreimb_total.to_numpy(float)
    presc = prescriptive_analysis(y, risk, prior_cost, phen, ph, p, n_boot=n_policy_boot, seed=seed, use_ilp=use_ilp)

    fp = p["prescriptive"]["fairness_policy"]
    eff, cst = scenario_vectors(phen, ph, p)
    budget, slots = budget_for(fp["cost"], fp["capacity"], len(y), p)
    masks, _ = select_all(risk, eff[fp["effect"]], cst[fp["cost"]], prior_cost, y, budget, True, use_ilp=False)
    fairness = subgroup_audit(y, risk, masks["optimized"], au, n_policy_boot, seed)
    fairness["policy"] = fp

    # Sensitivity: exclude members with 0 Part A months in 2009
    keep = lab.parta_months_2009.to_numpy() > 0
    b2, s2 = budget_for(fp["cost"], fp["capacity"], int(keep.sum()), p)
    m2, _ = select_all(risk[keep], eff[fp["effect"]][keep], cst[fp["cost"]][keep], prior_cost[keep], y[keep], b2, True)
    sens = {"excluded": int((~keep).sum()), "n": int(keep.sum()), "auroc": point_metrics(y[keep], risk[keep])["auroc"],
            "policy": {k: evaluate(v, y[keep], eff[fp["effect"]][keep], risk[keep], cst[fp["cost"]][keep]) for k, v in m2.items()}}

    # Pilot arithmetic from observed admission rates among targeted test members (highest risk, uniform cost)
    pc = p["pilot"]
    pilot = []
    for frac in p["prescriptive"]["capacity_fractions"]:
        b, slots = budget_for("uniform", frac, len(y), p)
        sel = select_all(risk, np.full(len(y), 0.2), np.ones(len(y)), prior_cost, y, b, True)[0]["highest_risk"]
        base = float(y[sel].mean())
        for rrr in pc["rrr_grid"]:
            n_arm = two_proportion_n(base, rrr, pc["alpha"], pc["power"])
            pilot.append({"capacity_fraction": frac, "control_admission_rate": base, "assumed_rrr": rrr,
                          "treated_admission_rate": base * (1 - rrr), "n_per_arm_individual": n_arm,
                          "cluster_design": {f"icc_{icc}": {"design_effect": design_effect(pc["cluster_size"], icc),
                                                            "n_per_arm": int(np.ceil(n_arm * design_effect(pc["cluster_size"], icc))),
                                                            "clusters_per_arm": int(np.ceil(n_arm * design_effect(pc["cluster_size"], icc) / pc["cluster_size"]))}
                                             for icc in pc["icc_grid"]}})

    for r in presc:
        lp = r["lp_upper_bound_planned_avoided"]
        r["optimized_relative_gap_to_lp_bound"] = (lp - r["point"]["optimized"]["planned_avoided"]) / lp if lp > 0 else 0.0
    out = {"phase": part, "created_at": datetime.now(timezone.utc).isoformat(), "freeze_commit": freeze_commit, "env": env_info(),
           "resamples": {"metrics": n_metric_boot, "policy": n_policy_boot}, "optimizer": "ILP (PuLP/CBC)" if use_ilp else "greedy",
           "synthetic_data_warning": "CMS DE-SynPUF is synthetic; no clinical or epidemiological conclusion is valid from these numbers.",
           "n_test": int(len(y)), "events_y_ip": int(y.sum()), "high_cost_threshold": thr, "targeting_model": chosen,
           "predictive": predictive, "phenotypes_test": by_phen, "prescriptive": presc, "fairness": fairness,
           "sensitivity_exclude_parta0_2009": sens, "pilot": pilot,
           "compute": {"seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb()}}
    write_json(out_path, out, exclusive=exclusive)
    print(part, "evaluation done", round(time.time() - t0), "s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dev", "dryrun", "freeze", "test"])
    {"dev": dev, "dryrun": dryrun, "freeze": freeze, "test": test}[ap.parse_args().phase]()
