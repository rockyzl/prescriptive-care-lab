"""Phenotyping: k-means on scaled 2008 chronic-condition + utilization profiles, fit on TRAIN only."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.features import Preprocessor


def cluster_matrix_fit(train_df, cfg):
    return Preprocessor(cfg["features"], cfg["log1p"]).fit(train_df)


def select_k(x_train, cfg, seed):
    rng = np.random.default_rng(seed)
    n = len(x_train)
    sub = rng.choice(n, min(cfg["silhouette_subsample"], n), replace=False)
    rows = []
    for k in cfg["k_range"]:
        ref = KMeans(k, n_init=cfg["kmeans_n_init"], random_state=seed).fit(x_train)
        ref_labels = ref.labels_
        sil = float(silhouette_score(x_train[sub], ref_labels[sub]))
        aris = []
        for s in cfg["stability_seeds"]:
            r = np.random.default_rng(s)
            idx = r.choice(n, int(cfg["stability_subsample_frac"] * n), replace=False)
            km = KMeans(k, n_init=cfg["kmeans_n_init"], random_state=s).fit(x_train[idx])
            aris.append(float(adjusted_rand_score(ref_labels, km.predict(x_train))))
        gmm = GaussianMixture(k, covariance_type=cfg["gmm_covariance"], random_state=seed, n_init=2).fit(x_train)
        rows.append({"k": k, "silhouette": sil, "stability_ari_mean": float(np.mean(aris)), "stability_ari_min": float(np.min(aris)),
                     "stability_aris": aris, "gmm_bic": float(gmm.bic(x_train)), "inertia": float(ref.inertia_)})
    return rows, choose_k(rows, cfg["stability_min_ari"])


def choose_k(rows, min_ari):
    ok = [r for r in rows if r["stability_ari_mean"] >= min_ari]
    if ok:
        best = max(ok, key=lambda r: (round(r["silhouette"], 6), -r["k"]))
        return best["k"], "highest silhouette among k with mean stability ARI >= %.2f" % min_ari
    best = max(rows, key=lambda r: (r["stability_ari_mean"], -r["k"]))
    return best["k"], "no k met the stability threshold; most stable k chosen"


def linear_by_rank(rank, n_groups, low, high):
    """Value for a phenotype whose risk rank is `rank` (0 = lowest train admission rate)."""
    if n_groups == 1:
        return (low + high) / 2
    return low + (high - low) * rank / (n_groups - 1)
