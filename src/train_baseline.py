#!/usr/bin/env python3
"""Task 2: fixed stratified split, dummy baseline, logistic regression, leakage demo."""
import json

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import (ALL_FEATURES, LEAKY_FEATURES, NUMERIC_FEATURES, TARGET,
                      build_preprocessor)

df = pd.read_csv("data/processed/customers_clean.csv", keep_default_na=False, na_values=[""])
train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print(f"train={len(train)} test={len(test)} "
      f"train_churn_rate={train[TARGET].mean():.3f} test_churn_rate={test[TARGET].mean():.3f}")


def evaluate(name, model, features):
    model.fit(train[features], train[TARGET])
    proba = model.predict_proba(test[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    res = {
        "model": name,
        "accuracy": round(float(accuracy_score(test[TARGET], pred)), 3),
        "roc_auc": round(float(roc_auc_score(test[TARGET], proba)), 3),
        "recall": round(float(recall_score(test[TARGET], pred, zero_division=0)), 3),
        "f1": round(float(f1_score(test[TARGET], pred, zero_division=0)), 3),
    }
    print(f"{name:20s} accuracy={res['accuracy']:.3f} roc_auc={res['roc_auc']:.3f} "
          f"recall={res['recall']:.3f} f1={res['f1']:.3f}")
    return res


results = [
    evaluate("dummy_most_frequent",
             Pipeline([("prep", build_preprocessor()),
                       ("clf", DummyClassifier(strategy="most_frequent"))]),
             ALL_FEATURES),
    evaluate("logreg",
             Pipeline([("prep", build_preprocessor()),
                       ("clf", LogisticRegression(max_iter=1000))]),
             ALL_FEATURES),
    evaluate("logreg_WITH_LEAK",
             Pipeline([("prep", build_preprocessor(numeric=NUMERIC_FEATURES + LEAKY_FEATURES)),
                       ("clf", LogisticRegression(max_iter=1000))]),
             ALL_FEATURES + LEAKY_FEATURES),
]
with open("reports/baseline.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved reports/baseline.json")
