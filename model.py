from __future__ import annotations
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def fit_segments(X: pd.DataFrame, k: int = 5):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
    ])
    labels = model.fit_predict(X)
    return model, labels

def profile_segments(X: pd.DataFrame, labels) -> pd.DataFrame:
    df = X.copy()
    df["segment"] = labels
    prof = df.groupby("segment").median(numeric_only=True)
    prof["count"] = df.groupby("segment").size()
    return prof.sort_values("count", ascending=False)
