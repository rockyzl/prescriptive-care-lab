"""Semi-synthetic program-effect estimation (protocol Addendum A).
The enrollment mechanism and the program effect are SIMULATED on top of observed 2009 outcomes."""
import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

PS_CLIP = (0.01, 0.99)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def solve_alpha(s, gamma, target=0.2):
    return brentq(lambda a: sigmoid(a + gamma * s).mean() - target, -30, 30)


def standardize(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def simulate(y0, s, gamma, rrr, rng, hidden_weight=False, target=0.2):
    """Return T, observed Y, Y1 and the true sample ATE.
    hidden_weight=True: enrollment also depends on a hidden severity signal u correlated with Y0 (not in X)."""
    y0 = np.asarray(y0, int)
    drive = s
    if hidden_weight:
        u = standardize(y0 + rng.normal(size=len(y0)))
        drive = (s + u) / np.sqrt(2)
    alpha = solve_alpha(drive, gamma, target)
    t = (rng.uniform(size=len(y0)) < sigmoid(alpha + gamma * drive)).astype(int)
    d = (rng.uniform(size=len(y0)) < rrr).astype(int)
    y1 = y0 * (1 - d)
    y = np.where(t == 1, y1, y0)
    return t, y, y1, float(np.mean(y1 - y0))


def _lr(x, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(x, y)


def estimate(x, t, y):
    """Naive, g-computation, Hajek IPW and AIPW estimates of the ATE (risk difference), with
    influence-function SEs for IPW/AIPW (propensity treated as known)."""
    n = len(y)
    e = np.clip(_lr(x, t).predict_proba(x)[:, 1], *PS_CLIP)
    m1 = _lr(x[t == 1], y[t == 1]).predict_proba(x)[:, 1]
    m0 = _lr(x[t == 0], y[t == 0]).predict_proba(x)[:, 1]
    naive = y[t == 1].mean() - y[t == 0].mean()
    gcomp = float(np.mean(m1 - m0))
    w1, w0 = t / e, (1 - t) / (1 - e)
    mu1, mu0 = np.sum(w1 * y) / np.sum(w1), np.sum(w0 * y) / np.sum(w0)
    ipw = mu1 - mu0
    phi_ipw = w1 * (y - mu1) / w1.mean() - w0 * (y - mu0) / w0.mean()
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    aipw = psi.mean()
    return {
        "naive": float(naive), "gcomp": gcomp, "ipw": float(ipw), "aipw": float(aipw),
        "ipw_se": float(phi_ipw.std(ddof=1) / np.sqrt(n)), "aipw_se": float(psi.std(ddof=1) / np.sqrt(n)),
        "_e": e,
    }


def smd(x, t, w=None):
    """Absolute standardized mean differences per column; denominator = unweighted pooled SD."""
    w = np.ones(len(t)) if w is None else w
    a, b = t == 1, t == 0
    m1 = np.average(x[a], axis=0, weights=w[a])
    m0 = np.average(x[b], axis=0, weights=w[b])
    sd = np.sqrt((x[a].var(axis=0) + x[b].var(axis=0)) / 2)
    sd = np.where(sd > 0, sd, 1.0)
    return np.abs(m1 - m0) / sd


def ess(w):
    return float(w.sum() ** 2 / np.sum(w ** 2))


def balance(x, t, e):
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    before, after = smd(x, t), smd(x, t, w)
    return {"max_abs_smd_before": float(before.max()), "n_smd_gt_0.1_before": int((before > 0.1).sum()),
            "max_abs_smd_after_ipw": float(after.max()), "n_smd_gt_0.1_after_ipw": int((after > 0.1).sum()),
            "ps_min": float(e.min()), "ps_max": float(e.max()),
            "ess_treated": ess(w[t == 1]), "ess_control": ess(w[t == 0]), "n_treated": int(t.sum()), "n_control": int((1 - t).sum()),
            "smd_before": before.tolist(), "smd_after_ipw": after.tolist()}
