#!/usr/bin/env python3
"""Task 3: 5-fold stratified CV comparison of candidate algorithms."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier  # noqa: F401
from sklearn.pipeline import Pipeline

from features import ALL_FEATURES, TARGET, build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[ALL_FEATURES], train[TARGET]

candidates = {
    "logreg": LogisticRegression(max_iter=1000),
    "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=42),
    "hist_gb": HistGradientBoostingClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=25, weights="distance"),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
           "recall": "recall", "precision": "precision", "f1": "f1"}

rows = []
for name, clf in candidates.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    s = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
    row = {"model": name}
    for m in scoring:
        row[f"{m}_mean"] = s[f"test_{m}"].mean()
        row[f"{m}_std"] = s[f"test_{m}"].std()
    row["fit_time_s"] = s["fit_time"].mean()
    rows.append(row)
    print(f"{name:16s} roc_auc={row['roc_auc_mean']:.3f}+/-{row['roc_auc_std']:.3f} "
          f"pr_auc={row['pr_auc_mean']:.3f} recall={row['recall_mean']:.3f} "
          f"precision={row['precision_mean']:.3f} f1={row['f1_mean']:.3f} "
          f"fit={row['fit_time_s']:.2f}s")

res = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
res.round(4).to_csv("reports/model_comparison.csv", index=False)
print("\nranking by ROC AUC:", list(res["model"]))
