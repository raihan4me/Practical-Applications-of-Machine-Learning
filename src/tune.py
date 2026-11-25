#!/usr/bin/env python3
"""Task 4: tune HistGradientBoosting and choose a cost-optimal decision threshold."""
import json

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from features import ALL_FEATURES, TARGET, build_preprocessor

COST_FN = 500.0
COST_CONTACT = 50.0


def business_cost(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    contacted = int(pred.sum())
    return fn * COST_FN + contacted * COST_CONTACT


def find_best_threshold(y_true, proba):
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = [business_cost(y_true, proba, t) for t in thresholds]
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])


def main():
    train = pd.read_csv("data/processed/train.csv", keep_default_na=False, na_values=[""])
    X, y = train[ALL_FEATURES], train[TARGET].to_numpy()

    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", HistGradientBoostingClassifier(random_state=42))])
    param_dist = {
        "clf__learning_rate": loguniform(0.02, 0.3),
        "clf__max_iter": randint(100, 400),
        "clf__max_leaf_nodes": randint(8, 48),
        "clf__min_samples_leaf": randint(10, 80),
        "clf__l2_regularization": loguniform(1e-3, 10.0),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=20, scoring="roc_auc", cv=cv,
                                random_state=42, n_jobs=1, verbose=1)
    search.fit(X, y)
    best_params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                   for k, v in search.best_params_.items()}
    print(f"best CV ROC AUC: {search.best_score_:.4f}")
    print(f"best params: {best_params}")

    oof = cross_val_predict(search.best_estimator_, X, y, cv=cv, method="predict_proba")[:, 1]
    thr, cost = find_best_threshold(y, oof)
    cost_05 = business_cost(y, oof, 0.5)
    cost_all = len(y) * COST_CONTACT
    cost_none = int(y.sum()) * COST_FN
    print(f"policy 'contact nobody'      cost: {cost_none:>10,.0f}")
    print(f"policy 'contact everybody'   cost: {cost_all:>10,.0f}")
    print(f"model @ threshold 0.50       cost: {cost_05:>10,.0f}")
    print(f"model @ threshold {thr:.2f}       cost: {cost:>10,.0f}  "
          f"(saving vs 0.50: {100 * (cost_05 - cost) / cost_05:.1f}%)")

    out = {
        "best_params": best_params,
        "cv_roc_auc": round(float(search.best_score_), 4),
        "threshold": thr,
        "oof_cost_at_threshold": cost,
        "oof_cost_at_0.5": cost_05,
        "cost_contact_all": cost_all,
        "cost_contact_none": cost_none,
        "cost_fn": COST_FN,
        "cost_contact": COST_CONTACT,
        "oof_mean_proba": round(float(oof.mean()), 4),
    }
    with open("reports/tuning.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved reports/tuning.json")


if __name__ == "__main__":
    main()
