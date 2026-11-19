"""Training-only preprocessing for the predictive models and the phenotyping step."""
import numpy as np
import pandas as pd

from src.data import FORBIDDEN_PREDICTORS

NON_FEATURES = {"split", "DESYNPUF_ID"}


def predictor_columns(feats: pd.DataFrame, protocol: dict) -> list:
    excluded = set(FORBIDDEN_PREDICTORS) | set(protocol["excluded_from_predictors"]) | NON_FEATURES
    return [c for c in feats.columns if c not in excluded]


class Preprocessor:
    """log1p on listed columns, then (optionally) standardize with TRAINING statistics only."""

    def __init__(self, columns, log1p_cols):
        self.columns = list(columns)
        self.log1p_cols = [c for c in log1p_cols if c in self.columns]

    def _log(self, df):
        x = df[self.columns].astype(float).copy()
        if (x[self.log1p_cols] < 0).any().any():
            raise ValueError("log1p column has negative values")
        x[self.log1p_cols] = np.log1p(x[self.log1p_cols])
        return x

    def fit(self, train_df):
        x = self._log(train_df)
        self.mean = x.mean().to_numpy()
        sd = x.std(ddof=0).to_numpy()
        self.scale = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, df, standardize=True):
        x = self._log(df).to_numpy()
        return (x - self.mean) / self.scale if standardize else x

    def to_dict(self):
        return {"columns": self.columns, "log1p": self.log1p_cols, "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["columns"], d["log1p"])
        p.mean, p.scale = np.array(d["mean"]), np.array(d["scale"])
        return p


def high_cost_threshold(train_cost, q):
    return float(np.quantile(np.asarray(train_cost, float), q))
