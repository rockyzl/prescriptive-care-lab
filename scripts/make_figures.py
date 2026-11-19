"""Charts + tables built ONLY from results/*.json.

  uv run python scripts/make_figures.py
writes reports/figures/*.png, results/tables.md and reports/summary.html
"""
import base64
import html
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# usage: make_figures.py [results_json] [out_dir]  (defaults: results/test_results.json, reports/)
RES_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "test_results.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports"
FIG = OUT / "figures"
DEV = json.loads((ROOT / "results" / "dev_results.json").read_text())
TEST = json.loads(RES_PATH.read_text())

# reference palette (dataviz skill, light mode), categorical slots in fixed order
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 150})

# Plain-language names written from the TRAIN profiles in results/dev_results.json (before the test run).
PHENOTYPE_NAMES = {0: "No recorded 2008 care", 5: "Light use, few conditions", 3: "Cardiometabolic multimorbidity",
                   2: "Cardiometabolic + arthritis", 1: "Cancer", 4: "ESRD / kidney disease",
                   7: "Recent hospital stays, complex", 6: "Stroke/TIA + hospital stays"}
POLICY_LABEL = {"random": "Random", "highest_cost": "Highest 2008 cost", "highest_risk": "Highest predicted risk",
                "optimized": "Optimized (ILP)", "oracle": "Oracle (knows outcome)"}
POLICY_COLOR = {"random": INK2, "highest_cost": ORANGE, "highest_risk": BLUE, "optimized": AQUA, "oracle": "#b9b8b2"}


def f3(x):
    return f"{x:.3f}"


def est_ci(d, fmt=f3):
    return f"{fmt(d['estimate'])} [{fmt(d['ci95'][0])}, {fmt(d['ci95'][1])}]"


def find(cost, frac, eff):
    for r in TEST["prescriptive"]:
        if r["cost_scenario"] == cost and abs(r["capacity_fraction"] - frac) < 1e-9 and r["effect_scenario"] == eff:
            return r
    raise KeyError((cost, frac, eff))


def fig_reliability():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, o, title in zip(axes, ["y_ip", "high_cost"], ["Any inpatient admission, 2009", "Top-decile 2009 cost"]):
        rel = TEST["predictive"][o]["reliability"]
        lim = max(max(b["mean_pred"], b["obs_rate"]) for m in rel.values() for b in m) * 1.08
        ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--", label="Perfect calibration")
        for m, c in [("logistic", ORANGE), ("lightgbm", BLUE)]:
            ax.plot([b["mean_pred"] for b in rel[m]], [b["obs_rate"] for b in rel[m]], marker="o", ms=5, lw=2, color=c,
                    label="Logistic regression" if m == "logistic" else "LightGBM")
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk (decile of members)", ylabel="Observed rate", title=title)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Calibration on held-out test members (synthetic data)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)


def fig_phenotypes():
    prof = {p["phenotype"]: p for p in DEV["clustering"]["profiles_train"]}
    test = {p["phenotype"]: p for p in TEST["phenotypes_test"]}
    order = sorted(prof, key=lambda c: prof[c]["risk_rank"])
    labels = [textwrap.fill(PHENOTYPE_NAMES[c], 15, break_long_words=False) + f"\n({prof[c]['share_train']*100:.0f}%)" for c in order]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], [prof[c]["train_admission_rate_2009"] * 100 for c in order], w, color=ORANGE, label="Train (fit set)")
    ax.bar([i + w / 2 + 0.01 for i in x], [test[c]["test_admission_rate_2009"] * 100 for c in order], w, color=BLUE, label="Test (held out)")
    for i, c in enumerate(order):
        ax.text(i + w / 2 + 0.01, test[c]["test_admission_rate_2009"] * 100 + 0.6, f"{test[c]['test_admission_rate_2009']*100:.0f}%",
                ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x), labels, fontsize=7.5)
    ax.set_ylabel("% with any 2009 admission")
    ax.set_title("Next-year admission rate by 2008 phenotype (ordered by train risk)", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "phenotypes.png")
    plt.close(fig)


def fig_policies():
    fracs = [0.02, 0.05, 0.10]
    pols = ["random", "highest_cost", "highest_risk", "oracle"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for j, pol in enumerate(pols):
        vals, lo, hi = [], [], []
        for f in fracs:
            r = find("uniform", f, "U20")
            v = r["point"][pol]["realized_avoided"]
            vals.append(v)
            c = r["bootstrap_ci95"].get(pol, {}).get("realized_avoided")
            lo.append(v - c[0] if c else 0)
            hi.append(c[1] - v if c else 0)
        xs = [i + (j - 1.5) * (w + 0.02) for i in range(len(fracs))]
        ax.bar(xs, vals, w, color=POLICY_COLOR[pol], label=POLICY_LABEL[pol] + (" = Optimized when effect is uniform" if pol == "highest_risk" else ""))
        if pol != "oracle":
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        for xx, v, h in zip(xs, vals, hi):
            ax.text(xx, v + h + 3, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(fracs)), [f"Contact {int(f*100)}% of members\n({find('uniform', f, 'U20')['capacity_slots']} people)" for f in fracs])
    ax.set_ylabel("Admissions avoided (assumed 20% effect)")
    ax.set_title("Who to contact: avoided admissions on test members, ASSUMED 20% risk reduction", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "policies.png")
    plt.close(fig)


def fig_sensitivity():
    effs = ["U10", "U20", "U30", "H_dec", "H_inc"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    for ax, cost, title in zip(axes, ["uniform", "hours"], ["Budget = number of contacts", "Budget = care-manager hours"]):
        for f, c in zip([0.02, 0.05, 0.10], [ORANGE, BLUE, AQUA]):
            d = [find(cost, f, e)["paired_differences"]["optimized-highest_risk"]["realized_avoided"] for e in effs]
            xs = [i + (f - 0.05) * 3 for i in range(len(effs))]
            ax.errorbar(xs, [x["estimate"] for x in d], yerr=[[x["estimate"] - x["ci95"][0] for x in d], [x["ci95"][1] - x["estimate"] for x in d]],
                        fmt="o", ms=6, color=c, ecolor=c, capsize=3, label=f"{int(f*100)}% capacity")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(effs)), ["10%\nuniform", "20%\nuniform", "30%\nuniform", "sickest\nleast helped", "sickest\nmost helped"], fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Optimized minus highest-risk\n(admissions avoided, 95% CI)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("The optimizer's advantage comes only from the assumed effect and cost differences", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png")
    plt.close(fig)


def fig_subgroups():
    g = TEST["fairness"]["groups"]
    keys = group_order(g)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    vals = [g[k]["capture_rate"] * 100 for k in keys]
    ax.barh(range(len(keys)), vals, color=BLUE, height=0.6)
    for i, k in enumerate(keys):
        lo, hi = g[k]["capture_rate_ci95"]
        ax.plot([lo * 100, hi * 100], [i, i], color=INK, lw=1)
        ax.text(hi * 100 + 0.4, i, f"{vals[i]:.1f}%  (n={g[k]['n']:,}, admitted {g[k]['admission_rate']*100:.0f}%)", va="center", fontsize=7.5, color=INK)
    ov = TEST["fairness"]["overall"]["capture_rate"] * 100
    ax.axvline(ov, color=ORANGE, lw=1.5, ls="--", label=f"All members {ov:.1f}%")
    ax.set_yticks(range(len(keys)), [k.replace("age_group=", "Age ").replace("sex=", "").replace("race=", "Race: ") for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of the group's 2009 admissions among contacted members (5% capacity)")
    ax.set_xlim(0, max(g[k]["capture_rate_ci95"][1] for k in keys) * 100 * 1.55)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Subgroup check: does targeting reach each group's admissions?", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "subgroups.png")
    plt.close(fig)


def group_order(g):
    ages = [f"age_group={a}" for a in ["<65", "65-74", "75-84", "85+"] if f"age_group={a}" in g]
    return ages + [k for k in g if not k.startswith("age_group=")]


def fig_shap():
    top = DEV["shap_val_subsample"][:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([t["feature"] for t in top], [t["mean_abs_shap"] for t in top], color=BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP| (log-odds), 5,000 validation members")
    ax.set_title("What the LightGBM risk model leans on (association, not cause)", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "shap.png")
    plt.close(fig)


def tables():
    L = []
    P = TEST["predictive"]
    L.append(f"## Predictive performance, held-out test (n={TEST['n_test']:,}; {TEST['events_y_ip']:,} admitted in 2009)\n")
    L.append("| Outcome | Model | AUROC [95% CI] | AUPRC [95% CI] | Brier [95% CI] | Calibration intercept | Calibration slope |")
    L.append("|---|---|---|---|---|---|---|")
    for o, name in [("y_ip", "Any admission 2009"), ("high_cost", "Top-decile cost 2009")]:
        for m, mn in [("logistic", "Logistic"), ("lightgbm", "LightGBM")]:
            d = P[o]["metrics"][m]
            L.append(f"| {name} (prevalence {P[o]['prevalence']*100:.1f}%) | {mn} | {est_ci(d['auroc'])} | {est_ci(d['auprc'])} | {est_ci(d['brier'])} | "
                     f"{est_ci(d['cal_intercept'], lambda v: f'{v:.2f}')} | {est_ci(d['cal_slope'], lambda v: f'{v:.2f}')} |")
    L.append("")
    L.append("| Outcome | LightGBM − logistic: ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔBrier [95% CI] |")
    L.append("|---|---|---|---|")
    for o in ["y_ip", "high_cost"]:
        d = P[o]["paired_differences"]["lightgbm-logistic"]
        L.append(f"| {o} | {est_ci(d['auroc'])} | {est_ci(d['auprc'])} | {est_ci(d['brier'], lambda v: f'{v:.4f}')} |")
    L.append("")
    L.append(f"## Phenotypes (k = {DEV['clustering']['chosen_k']}, fit on train)\n")
    L.append("| Phenotype | Train share | Train admission rate | Test n | Test admission rate | Test mean predicted risk |")
    L.append("|---|---|---|---|---|---|")
    test = {p["phenotype"]: p for p in TEST["phenotypes_test"]}
    for p in sorted(DEV["clustering"]["profiles_train"], key=lambda p: p["risk_rank"]):
        t = test[p["phenotype"]]
        L.append(f"| P{p['phenotype']} {PHENOTYPE_NAMES[p['phenotype']]} | {p['share_train']*100:.1f}% | {p['train_admission_rate_2009']*100:.1f}% | {t['n_test']:,} | "
                 f"{t['test_admission_rate_2009']*100:.1f}% | {t['test_mean_predicted_risk']*100:.1f}% |")
    L.append("")
    L.append("## Targeting on test: realized avoided admissions = Σ observed admission × ASSUMED risk reduction\n")
    L.append("| Budget | Capacity | Effect assumption | Random (mean) | Highest cost [95% CI] | Highest risk [95% CI] | Optimized [95% CI] | Oracle | Optimized − highest risk [95% CI] | Highest risk − highest cost [95% CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in TEST["prescriptive"]:
        pt, cis, df = r["point"], r["bootstrap_ci95"], r["paired_differences"]
        cell = lambda k: f"{pt[k]['realized_avoided']:.1f} [{cis[k]['realized_avoided'][0]:.1f}, {cis[k]['realized_avoided'][1]:.1f}]"
        L.append(f"| {r['cost_scenario']} | {int(r['capacity_fraction']*100)}% ({r['capacity_slots']}) | {r['effect_scenario']} | {pt['random']['realized_avoided']:.1f} | "
                 f"{cell('highest_cost')} | {cell('highest_risk')} | {cell('optimized')} | {pt['oracle']['realized_avoided']:.1f} | "
                 f"{est_ci(df['optimized-highest_risk']['realized_avoided'], lambda v: f'{v:.1f}')} | {est_ci(df['highest_risk-highest_cost']['realized_avoided'], lambda v: f'{v:.1f}')} |")
    L.append("")
    L.append("## Admissions captured (assumption-free): members contacted who were admitted in 2009, uniform budget\n")
    L.append("| Capacity | Random (mean) | Highest cost [95% CI] | Highest risk [95% CI] | Risk − cost [95% CI] | PPV highest risk |")
    L.append("|---|---|---|---|---|---|")
    for f in [0.02, 0.05, 0.10]:
        r = find("uniform", f, "U20")
        pt, cis = r["point"], r["bootstrap_ci95"]
        d = r["paired_differences"]["highest_risk-highest_cost"]["admissions_captured"]
        L.append(f"| {int(f*100)}% ({r['capacity_slots']}) | {pt['random']['admissions_captured']:.1f} | {pt['highest_cost']['admissions_captured']:.0f} "
                 f"[{cis['highest_cost']['admissions_captured'][0]:.0f}, {cis['highest_cost']['admissions_captured'][1]:.0f}] | {pt['highest_risk']['admissions_captured']:.0f} "
                 f"[{cis['highest_risk']['admissions_captured'][0]:.0f}, {cis['highest_risk']['admissions_captured'][1]:.0f}] | {est_ci(d, lambda v: f'{v:.0f}')} | {pt['highest_risk']['ppv']*100:.1f}% |")
    L.append("")
    L.append("## Subgroup audit (5% capacity, uniform 20% effect, uniform cost)\n")
    L.append("| Group | n | Admission rate | Mean predicted | AUROC [95% CI] | Selection rate | Capture rate [95% CI] | Flag |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in group_order(TEST["fairness"]["groups"]):
        g = TEST["fairness"]["groups"][k]
        L.append(f"| {k} | {g['n']:,} | {g['admission_rate']*100:.1f}% | {g['mean_predicted']*100:.1f}% | {g['auroc']:.3f} [{g['auroc_ci95'][0]:.3f}, {g['auroc_ci95'][1]:.3f}] | "
                 f"{g['selection_rate']*100:.1f}% | {g['capture_rate']*100:.1f}% [{g['capture_rate_ci95'][0]*100:.1f}, {g['capture_rate_ci95'][1]*100:.1f}] | {'yes' if g['flag_capture_gap_gt_5pp'] else ''} |")
    ov = TEST["fairness"]["overall"]
    L.append(f"| all | {TEST['n_test']:,} | {ov['admission_rate']*100:.1f}% | | | {ov['selection_rate']*100:.1f}% | {ov['capture_rate']*100:.1f}% [{ov['capture_rate_ci95'][0]*100:.1f}, {ov['capture_rate_ci95'][1]*100:.1f}] | |")
    L.append("")
    L.append("## Pilot sizing (planning arithmetic; two-sided α 0.05, power 0.8)\n")
    L.append("| Capacity | Control admission rate (observed, targeted test members) | Assumed RRR | n per arm (individual) | n per arm, cluster ICC 0.01 | n per arm, ICC 0.05 |")
    L.append("|---|---|---|---|---|---|")
    for p in TEST["pilot"]:
        cd = p["cluster_design"]
        L.append(f"| {int(p['capacity_fraction']*100)}% | {p['control_admission_rate']*100:.1f}% | {int(p['assumed_rrr']*100)}% | {p['n_per_arm_individual']:,} | "
                 f"{cd['icc_0.01']['n_per_arm']:,} ({cd['icc_0.01']['clusters_per_arm']} clusters of 50) | {cd['icc_0.05']['n_per_arm']:,} ({cd['icc_0.05']['clusters_per_arm']} clusters) |")
    return "\n".join(L) + "\n"


def summary_html(tables_md):
    def img(name):
        return "data:image/png;base64," + base64.b64encode((FIG / name).read_bytes()).decode()
    P = TEST["predictive"]["y_ip"]["metrics"][TEST["targeting_model"]]
    r5 = find("uniform", 0.05, "U20")
    k = DEV["clustering"]["chosen_k"]
    tiles = [
        (f"{P['auroc']['estimate']:.2f}", f"AUROC for next-year admission (95% CI {P['auroc']['ci95'][0]:.2f}–{P['auroc']['ci95'][1]:.2f})"),
        (f"{k}", "stable phenotypes from 2008 conditions and use"),
        (f"{r5['point']['highest_risk']['admissions_captured']:.0f} vs {r5['point']['random']['admissions_captured']:.0f}",
         f"2009 admissions among {r5['capacity_slots']} contacted members: risk-targeted vs random"),
    ]
    tile_html = "".join(f'<div class="tile"><div class="num">{html.escape(a)}</div><div class="lab">{html.escape(b)}</div></div>' for a, b in tiles)
    md_tables = html.escape(tables_md)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Care Targeting Prototype</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --warn-bg:#fff4e0; --warn-ink:#6b4500; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#3a3a37; --warn-bg:#3a2c10; --warn-ink:#f5d99a; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#3a3a37; --warn-bg:#3a2c10; --warn-ink:#f5d99a; }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:960px; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; }} h2 {{ font-size:1.2rem; margin-top:2rem; }}
.sub {{ color:var(--ink2); margin:0 0 16px; }}
.warn {{ background:var(--warn-bg); color:var(--warn-ink); padding:12px 14px; border-radius:8px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:16px 0; }}
.tile {{ border:1px solid var(--line); border-radius:8px; padding:12px; }}
.num {{ font-size:1.8rem; font-weight:600; }} .lab {{ color:var(--ink2); font-size:.9rem; }}
img {{ max-width:100%; height:auto; border-radius:6px; background:#fcfcfb; }}
pre {{ overflow-x:auto; font-size:12px; border:1px solid var(--line); padding:10px; border-radius:6px; }}
details summary {{ cursor:pointer; color:var(--ink2); }}
</style></head><body><main>
<h1>Care-management targeting prototype</h1>
<p class="sub">Predict next-year admission, group members into phenotypes, then choose whom to contact with limited capacity. Research prototype on CMS synthetic Medicare claims (DE-SynPUF Sample 1).</p>
<p class="warn"><strong>Synthetic data, assumed effects.</strong> CMS DE-SynPUF is synthetic. No clinical or epidemiological conclusion is valid from these numbers. The intervention effect (10–30% fewer admissions) is an <em>assumption</em>, not a measurement. Only a randomized pilot can measure it.</p>
<div class="tiles">{tile_html}</div>
<h2>1. Whom to contact</h2><img src="{img('policies.png')}" alt="Bar chart of avoided admissions by targeting policy and capacity">
<h2>2. Do the assumptions change the answer?</h2><img src="{img('sensitivity.png')}" alt="Optimized minus highest-risk difference across effect assumptions">
<h2>3. Phenotypes</h2><img src="{img('phenotypes.png')}" alt="Admission rate by phenotype">
<h2>4. Are the risk numbers honest? (calibration)</h2><img src="{img('calibration.png')}" alt="Reliability curves">
<h2>5. Subgroup check</h2><img src="{img('subgroups.png')}" alt="Capture rate by subgroup">
<h2>6. What drives the risk score</h2><img src="{img('shap.png')}" alt="Top SHAP features">
<details><summary>All result tables (generated from results/*.json)</summary><pre>{md_tables}</pre></details>
<p class="sub">Generated by scripts/make_figures.py from results/dev_results.json and results/test_results.json. Not a clinical tool.</p>
</main></body></html>
"""


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_reliability()
    fig_phenotypes()
    fig_policies()
    fig_sensitivity()
    fig_subgroups()
    fig_shap()
    t = tables()
    tables_path = ROOT / "results" / "tables.md" if OUT == ROOT / "reports" else OUT / "tables.md"
    tables_path.write_text(f"# Result tables (generated from {RES_PATH.name} and dev_results.json)\n\n" + t)
    (OUT / "summary.html").write_text(summary_html(t))
    print("wrote figures, tables and summary.html to", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
