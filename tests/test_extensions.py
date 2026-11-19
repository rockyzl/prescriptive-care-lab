import numpy as np
import pandas as pd
import pytest

from src.causal import balance, estimate, simulate, sigmoid, solve_alpha, standardize
from src.forecast import backtest, forecast, in_window, series_from_claims, summarize


# ------------------------------------------------------------------ causal (Addendum A)
def test_enrollment_rate_hits_target():
    s = np.random.default_rng(0).normal(size=5000)
    for g in (0.0, 1.0, 2.0):
        assert sigmoid(solve_alpha(s, g, 0.2) + g * s).mean() == pytest.approx(0.2, abs=1e-6)


def test_simulated_potential_outcomes():
    rng = np.random.default_rng(1)
    y0 = (rng.uniform(size=20000) < 0.2).astype(int)
    s = standardize(y0 + rng.normal(size=20000))
    t, y, y1, ate = simulate(y0, s, 1.0, 0.0, np.random.default_rng(2))
    assert ate == 0 and (y == y0).all()
    t, y, y1, ate = simulate(y0, s, 1.0, 0.3, np.random.default_rng(2))
    assert (y1 <= y0).all()
    assert (y[t == 0] == y0[t == 0]).all() and (y[t == 1] == y1[t == 1]).all()
    assert ate == pytest.approx(-0.3 * y0.mean(), abs=0.01)


def _toy(n=20000, seed=3, effect=-0.05):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    e = sigmoid(-1.4 + 1.2 * x[:, 0])
    t = (rng.uniform(size=n) < e).astype(int)
    p0 = sigmoid(-1.5 + 1.0 * x[:, 0] + 0.5 * x[:, 1])
    y0 = (rng.uniform(size=n) < p0).astype(int)
    y1 = (rng.uniform(size=n) < np.clip(p0 + effect, 0, 1)).astype(int)
    y = np.where(t == 1, y1, y0)
    return x, t, y, float(np.mean(y1 - y0))


def test_adjusted_estimators_recover_known_effect_and_naive_is_biased():
    x, t, y, ate = _toy()
    est = estimate(x, t, y)
    assert abs(est["naive"] - ate) > 0.05          # confounding by x0 biases the naive contrast
    for k in ("gcomp", "ipw", "aipw"):
        assert abs(est[k] - ate) < 0.02
    assert abs(est["aipw"] - ate) < 3 * est["aipw_se"]


def test_ipw_weighting_improves_balance():
    x, t, y, _ = _toy()
    est = estimate(x, t, y)
    b = balance(x, t, est["_e"])
    assert b["max_abs_smd_before"] > 0.3 and b["max_abs_smd_after_ipw"] < 0.05


# ------------------------------------------------------------------ forecasting (Addendum B)
def test_series_counts_distinct_claims_by_admission_month():
    ip = pd.DataFrame({"DESYNPUF_ID": ["a", "a", "b", "c"], "CLM_ID": ["1", "1", "2", "3"],
                       "CLM_ADMSN_DT": [20080305, 20080305, np.nan, 20080420], "CLM_FROM_DT": [20080305, 20080305, 20080331, 20080420],
                       "CLM_PMT_AMT": [100.0, 100.0, 50.0, 10.0]})
    s = series_from_claims(ip)
    assert s.admissions.to_dict() == {200803: 2, 200804: 1}
    assert s.payments.to_dict() == {200803: 150.0, 200804: 10.0}


def test_window_excludes_rampup_and_2010():
    idx = [200801, 200802, 200803, 200912, 201001]
    assert in_window(idx).tolist() == [False, False, True, True, False]


def test_simple_forecasts():
    y = np.arange(1, 25, dtype=float)  # 24 months, linear
    assert forecast(y, 3, "naive")["point"].tolist() == [24, 24, 24]
    assert forecast(y, 3, "drift")["point"].tolist() == [25, 26, 27]
    assert forecast(y, 2, "seasonal_naive")["point"].tolist() == [13, 14]
    lo, hi = forecast(y + np.random.default_rng(0).normal(size=24), 3, "naive")[0.8]
    assert (hi - lo)[2] > (hi - lo)[0]  # intervals widen with horizon


def test_backtest_uses_only_past_data():
    rng = np.random.default_rng(4)
    y = 100 + rng.normal(size=22).cumsum()
    a = backtest(y, ["naive", "drift", "seasonal_naive"])
    y2 = y.copy()
    y2[15:] += 1000  # change the future after origin 15
    b = backtest(y2, ["naive", "drift", "seasonal_naive"])
    early = a.origin_len <= 15
    pd.testing.assert_series_equal(a.loc[early, "point"], b.loc[early, "point"])
    assert (a.target_pos >= a.origin_len).all()
    assert set(a.groupby(["method", "h"]).size()) == {10, 8}


def test_coverage_summary():
    bt = pd.DataFrame({"method": ["m"] * 4, "h": [1] * 4, "actual": [1.0, 2, 3, 4], "point": [1.0, 2, 3, 4],
                       "lo80": [0.0, 0, 0, 5], "hi80": [9.0, 9, 9, 9], "lo95": [0.0] * 4, "hi95": [9.0] * 4})
    s = summarize(bt, 1.0)["m|h=1"]
    assert s["coverage80"] == 0.75 and s["coverage95"] == 1.0 and s["mae"] == 0
