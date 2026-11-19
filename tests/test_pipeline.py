import json

import numpy as np
import pandas as pd
import pytest

from src.data import (DERIVED, FORBIDDEN_PREDICTORS, MEMBERSHIP, ROOT, admissions_in_year, claim_block_features, feature_window,
                      icd9_chapter, split_for)
from src.features import Preprocessor, high_cost_threshold, predictor_columns
from src.models import calibration_intercept_slope
from src.prescriptive import greedy, lp_upper_bound, random_policy, rank_policy, solve_ilp, take_in_order, two_proportion_n

HAS_DATA = (DERIVED / "features.parquet").exists() and MEMBERSHIP.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="run `python -m src.data` first")
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())


# ------------------------------------------------------------------ split integrity
def test_split_is_deterministic_disjoint_and_id_only():
    ids = [f"{i:016X}" for i in range(20000)]
    a, b = split_for(ids), split_for(list(reversed(ids)))
    assert a == b  # independent of row order
    s = pd.Series(a)
    frac = s.value_counts(normalize=True)
    assert abs(frac["train"] - 0.6) < 0.015 and abs(frac["val"] - 0.2) < 0.015 and abs(frac["test"] - 0.2) < 0.015
    with pytest.raises(ValueError):
        split_for(ids + ids[:1])


@needs_data
def test_saved_membership_matches_rule_and_label_files():
    membership = json.loads(MEMBERSHIP.read_text())
    feats = pd.read_parquet(DERIVED / "features.parquet")
    assert feats.DESYNPUF_ID.is_unique
    assert split_for(feats.DESYNPUF_ID) == membership
    seen = set()
    for part in ["train", "val", "test"]:
        lab = pd.read_parquet(DERIVED / f"labels_{part}.parquet")
        assert lab.DESYNPUF_ID.is_unique
        assert (lab.DESYNPUF_ID.map(membership) == part).all()
        assert not seen & set(lab.DESYNPUF_ID)
        seen |= set(lab.DESYNPUF_ID)
    assert seen == set(feats.DESYNPUF_ID)


@needs_data
def test_dev_loader_never_returns_test_labels():
    from src.run import load

    _, _, labels = load(["train", "val"])
    assert set(labels) == {"train", "val"}
    test_ids = set(pd.read_parquet(DERIVED / "labels_test.parquet").DESYNPUF_ID)
    for lab in labels.values():
        assert not test_ids & set(lab.index)


# ------------------------------------------------------------------ no future leakage
def _claims(rows):
    cols = ["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_THRU_DT", "DX1", "PAY"]
    return pd.DataFrame(rows, columns=cols)


def test_feature_window_keeps_only_claims_fully_inside_2008():
    df = _claims([
        ["A", "1", 20080105, 20080110, "4280", 100.0],   # inside 2008 -> kept
        ["A", "2", 20081228, 20090103, "4280", 500.0],   # spans into 2009 -> dropped
        ["A", "3", 20090301, 20090302, "4912", 900.0],   # 2009 -> dropped
        ["B", "4", 20071230, 20080102, "25000", 50.0],   # spans from 2007 -> dropped
        ["B", "5", 20080601, 20080601, "25000", 20.0],   # inside -> kept
    ])
    w = feature_window(df)
    assert w.CLM_ID.tolist() == ["1", "5"]


def test_changing_2009_claims_cannot_change_features():
    base = _claims([["A", "1", 20080105, 20080110, "4280", 100.0], ["A", "3", 20090301, 20090302, "4912", 900.0]])
    altered = base.copy()
    altered.loc[1, ["DX1", "PAY"]] = ["E8889", 99999.0]
    altered = pd.concat([altered, _claims([["A", "9", 20091201, 20091231, "V5841", 7.0]])])
    f1, l1 = claim_block_features(feature_window(base), "x", ["DX1"], ["PAY"])
    f2, l2 = claim_block_features(feature_window(altered), "x", ["DX1"], ["PAY"])
    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_frame_equal(l1.reset_index(drop=True), l2.reset_index(drop=True))
    assert f1.loc["A", "x_claims"] == 1 and f1.loc["A", "x_paid"] == 100.0


def test_outcome_counts_only_2009_admissions():
    ip = pd.DataFrame({"DESYNPUF_ID": ["A", "A", "A", "B", "C"], "CLM_ID": ["1", "1", "2", "3", "4"],
                       "CLM_ADMSN_DT": [20090105, 20090105, 20081230, np.nan, 20100101],
                       "CLM_FROM_DT": [20090105, 20090105, 20081230, 20090704, 20100101]})
    n = admissions_in_year(ip, 2009)
    assert n.to_dict() == {"A": 1, "B": 1}  # duplicate segment collapsed; 2008-admitted stay is not an outcome


def test_icd9_chapters():
    assert icd9_chapter("4280") == "circulatory"
    assert icd9_chapter("25000") == "endocrine_metabolic"
    assert icd9_chapter("V5841") == "v_codes" and icd9_chapter("E9330") == "e_codes"
    assert icd9_chapter("OTHER") is None and icd9_chapter(None) is None


@needs_data
def test_no_forbidden_or_outcome_column_reaches_predictors():
    feats = pd.read_parquet(DERIVED / "features.parquet")
    cols = predictor_columns(feats, PROTOCOL)
    assert not set(cols) & FORBIDDEN_PREDICTORS
    assert "race" not in feats.columns and "split" not in cols
    for bad in ["y_ip", "cost_2009", "n_adm_2009", "death_2009", "parta_months_2009", "high_cost"]:
        assert bad not in feats.columns
    assert len(cols) == 54


# ------------------------------------------------------------------ training-only preprocessing
def test_preprocessor_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"a": rng.poisson(3, 500).astype(float), "b": rng.normal(size=500)})
    ev = pd.DataFrame({"a": [1e6, 0.0], "b": [1e3, -1e3]})
    p1 = Preprocessor(["a", "b"], ["a"]).fit(tr)
    p2 = Preprocessor(["a", "b"], ["a"]).fit(tr)
    p2.transform(ev)  # transforming eval rows must not refit anything
    assert np.allclose(p1.mean, p2.mean) and np.allclose(p1.scale, p2.scale)
    assert np.allclose(p1.transform(tr).mean(0), 0, atol=1e-9)
    assert p1.mean[0] == pytest.approx(np.log1p(tr.a).mean())


def test_high_cost_threshold_is_training_quantile():
    assert high_cost_threshold([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == pytest.approx(9.1)


@needs_data
def test_saved_artefacts_were_fit_on_train_only():
    meta_path, pre_path = ROOT / "models" / "dev_meta.json", ROOT / "models" / "preprocessor.json"
    if not meta_path.exists():
        pytest.skip("run `python -m src.run dev` first")
    feats = pd.read_parquet(DERIVED / "features.parquet").set_index("DESYNPUF_ID")
    train = pd.read_parquet(DERIVED / "labels_train.parquet")
    meta = json.loads(meta_path.read_text())
    assert meta["high_cost_threshold"] == pytest.approx(high_cost_threshold(train.cost_2009, PROTOCOL["high_cost_quantile"]))
    saved = Preprocessor.from_dict(json.loads(pre_path.read_text()))
    refit = Preprocessor(saved.columns, saved.log1p_cols).fit(feats.loc[train.DESYNPUF_ID])
    assert np.allclose(saved.mean, refit.mean) and np.allclose(saved.scale, refit.scale)
    ph = json.loads((ROOT / "models" / "phenotypes.json").read_text())
    cpre = Preprocessor.from_dict(ph["preprocessor"])
    crefit = Preprocessor(cpre.columns, cpre.log1p_cols).fit(feats.loc[train.DESYNPUF_ID])
    assert np.allclose(cpre.mean, crefit.mean)


# ------------------------------------------------------------------ optimizer respects capacity
def _naive_take(order, cost, budget):
    mask, used = np.zeros(len(cost), bool), 0.0
    for i in order:
        if used + cost[i] <= budget + 1e-9:
            mask[i] = True
            used += cost[i]
    return mask


def test_take_in_order_matches_sequential_rule():
    rng = np.random.default_rng(1)
    for _ in range(50):
        cost = rng.choice([4.0, 5.0, 6.5, 8.0], 300)
        order = rng.permutation(300)
        budget = float(rng.uniform(10, 900))
        assert (take_in_order(order, cost, budget) == _naive_take(order, cost, budget)).all()


def test_uniform_cost_ilp_equals_greedy_and_respects_capacity():
    rng = np.random.default_rng(2)
    v = rng.uniform(0, 0.3, 2000)
    w = np.ones(2000)
    for budget in [1, 40, 100, 400]:
        ilp, status = solve_ilp(v, w, budget)
        g = greedy(v, w, budget)
        assert status == "Optimal"
        assert ilp.sum() <= budget and g.sum() == budget
        assert v[ilp].sum() == pytest.approx(v[g].sum(), rel=1e-9)


def test_knapsack_ilp_respects_budget_and_beats_or_ties_greedy():
    rng = np.random.default_rng(3)
    v = rng.beta(1, 5, 5000) * 0.3
    w = rng.choice([4.0, 5.0, 6.0, 7.0, 8.0], 5000)
    for budget in [50.0, 1203.0, 3000.0]:
        ilp, status = solve_ilp(v, w, budget)
        g = greedy(v, w, budget)
        assert status == "Optimal"
        assert w[ilp].sum() <= budget + 1e-6 and w[g].sum() <= budget + 1e-6
        assert v[ilp].sum() >= v[g].sum() - 1e-7
        assert v[ilp].sum() <= lp_upper_bound(v, w, budget) + 1e-9
        assert v[ilp].sum() >= lp_upper_bound(v, w, budget) - v.max()  # knapsack integrality gap < one item


def test_all_policies_respect_capacity():
    rng = np.random.default_rng(4)
    n = 1000
    score = rng.uniform(size=n)
    for cost in [np.ones(n), rng.choice([4.0, 6.0, 8.0], n)]:
        for budget in [20.0, 137.0]:
            for mask in [rank_policy(score, cost, budget), greedy(score, cost, budget), random_policy(n, cost, budget, rng)]:
                assert cost[mask].sum() <= budget + 1e-9


# ------------------------------------------------------------------ metrics and pilot arithmetic
def test_calibration_of_perfectly_calibrated_predictions():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.6, 200000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.02 and abs(b - 1) < 0.03


def test_two_proportion_sample_size_reference_value():
    # p1 = 0.5 vs p2 = 0.4 (RRR 20%), alpha 0.05, power 0.8 -> 388 per arm (pooled-variance formula)
    assert two_proportion_n(0.5, 0.2) == 388
