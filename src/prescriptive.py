"""Budgeted care-management targeting: ILP (PuLP/CBC), greedy, naive policies, realized evaluation,
bootstrap with re-selection, and pilot sample-size arithmetic."""
import math

import numpy as np
import pulp
from scipy.stats import norm


def solve_ilp(value, cost, budget, time_limit=120, gap_rel=1e-5):
    """max sum x_i*value_i  s.t. sum x_i*cost_i <= budget, x binary. Returns boolean mask.
    CBC stops at a relative MIP gap of `gap_rel`; `lp_upper_bound` reports the true remaining gap."""
    value, cost = np.asarray(value, float), np.asarray(cost, float)
    cand = np.where(value > 0)[0]
    prob = pulp.LpProblem("care_management_targeting", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", cand.tolist(), cat="Binary")
    prob += pulp.lpSum(float(value[i]) * x[i] for i in cand)
    prob += pulp.lpSum(float(cost[i]) * x[i] for i in cand) <= float(budget)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap_rel))
    mask = np.zeros(len(value), bool)
    for i in cand:
        if (x[i].value() or 0) > 0.5:
            mask[i] = True
    return mask, pulp.LpStatus[status]


def lp_upper_bound(value, cost, budget):
    """Fractional-knapsack (LP relaxation) optimum: an upper bound on any 0/1 selection."""
    value, cost = np.asarray(value, float), np.asarray(cost, float)
    keep = value > 0
    v, w = value[keep], cost[keep]
    order = np.argsort(-(v / w), kind="stable")
    cw = np.cumsum(w[order])
    k = int(np.searchsorted(cw, budget, side="right"))
    total = v[order[:k]].sum()
    if k < len(order):
        total += v[order[k]] * (budget - (cw[k - 1] if k > 0 else 0.0)) / w[order[k]]
    return float(total)


def take_in_order(order, cost, budget):
    """Walk `order`, contacting each member whose cost still fits in the remaining budget
    (vectorized prefix, then a short loop over the tail; identical to the sequential rule)."""
    order = np.asarray(order)
    cost = np.asarray(cost, float)
    mask = np.zeros(len(cost), bool)
    cs = np.cumsum(cost[order])
    k = int(np.searchsorted(cs, budget + 1e-9, side="right"))
    mask[order[:k]] = True
    remaining = budget - (cs[k - 1] if k > 0 else 0.0)
    rest = order[k:]
    while remaining > 1e-9 and len(rest):
        fits = np.nonzero(cost[rest] <= remaining + 1e-9)[0]
        if not len(fits):
            break
        i = rest[fits[0]]
        mask[i] = True
        remaining -= cost[i]
        rest = rest[fits[0] + 1:]
    return mask


def greedy(value, cost, budget):
    """Greedy by value/cost ratio. Exactly optimal when all costs are equal (budget = head-count)."""
    value, cost = np.asarray(value, float), np.asarray(cost, float)
    order = np.lexsort((np.arange(len(value)), -(value / cost)))
    return take_in_order(order, cost, budget)


def rank_policy(score, cost, budget):
    order = np.lexsort((np.arange(len(score)), -np.asarray(score, float)))
    return take_in_order(order, cost, budget)


def random_policy(n, cost, budget, rng):
    return take_in_order(rng.permutation(n), cost, budget)


def evaluate(mask, y, r, p, cost):
    return {
        "contacted": int(mask.sum()),
        "cost_used": float(np.asarray(cost)[mask].sum()),
        "admissions_captured": float(np.asarray(y)[mask].sum()),
        "realized_avoided": float((np.asarray(y) * r)[mask].sum()),
        "planned_avoided": float((np.asarray(p) * r)[mask].sum()),
        "ppv": float(np.asarray(y)[mask].mean()) if mask.any() else float("nan"),
    }


def select_all(p, r, cost, prior_cost, y, budget, uniform_cost, use_ilp=False):
    """Masks for the deterministic policies. Optimized uses the ILP when use_ilp else greedy
    (greedy == ILP for uniform cost)."""
    masks = {
        "highest_cost": rank_policy(prior_cost, cost, budget),
        "highest_risk": rank_policy(p, cost, budget),
        "oracle": greedy(y * r + 1e-12 * p, cost, budget),
    }
    status = None
    if use_ilp:
        masks["optimized"], status = solve_ilp(p * r, cost, budget)
    else:
        masks["optimized"] = greedy(p * r, cost, budget)
    return masks, status


# ----------------------------------------------------------------------------- pilot arithmetic
def two_proportion_n(p_control, rrr, alpha=0.05, power=0.8):
    """Per-arm n for a two-sided two-proportion z-test (pooled variance under H0)."""
    p1, p2 = p_control, p_control * (1 - rrr)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def design_effect(cluster_size, icc):
    return 1 + (cluster_size - 1) * icc
