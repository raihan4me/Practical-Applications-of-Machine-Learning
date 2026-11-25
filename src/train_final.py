#!/usr/bin/env python3
"""Task 5: train final model, evaluate on hold-out, explain, and package."""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from features import (ALL_FEATURES, CATEGORICAL_FEATURES, ID_COLUMN, LEAKY_FEATURES,
                      NUMERIC_FEATURES, TARGET, build_preprocessor)
from tune import COST_CONTACT, COST_FN, business_cost

QUALITY_GATE_AUC = 0.75
MODEL_VERSION = "1.0.0"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tuning = json.load(open("reports/tuning.json"))
    thr = float(tuning["threshold"])
    train = pd.read_csv("data/processed/train.csv", keep_default_na=False, na_values=[""])
    test = pd.read_csv("data/processed/test.csv", keep_default_na=False, na_values=[""])
    y_test = test[TARGET].to_numpy()

    model = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(random_state=42, **tuning["best_params"])),
    ])
    model.fit(train[ALL_FEATURES], train[TARGET])

    proba = model.predict_proba(test[ALL_FEATURES])[:, 1]
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "precision_at_threshold": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "contact_rate_at_threshold": round(float(pred.mean()), 4),
        "business_cost_at_threshold": business_cost(y_test, proba, thr),
        "business_cost_at_0.5": business_cost(y_test, proba, 0.5),
    }
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    print(f"== Hold-out test metrics (threshold={thr:.2f}) ==")
    for k, v in metrics.items():
        print(f"{k:28s} {v}")
    print(f"confusion matrix             {confusion}")

    if metrics["roc_auc"] < QUALITY_GATE_AUC:
        print(f"QUALITY GATE FAILED: roc_auc {metrics['roc_auc']} < {QUALITY_GATE_AUC}")
        sys.exit(1)
    print(f"quality gate passed (roc_auc >= {QUALITY_GATE_AUC})")

    print("\n== Subgroup slice: senior_citizen ==")
    slices = {}
    for g, part in test.assign(pred=pred).groupby("senior_citizen"):
        slices[f"senior_citizen={g}"] = {
            "n": int(len(part)),
            "churn_rate": round(float(part[TARGET].mean()), 3),
            "recall": round(float(recall_score(part[TARGET], part["pred"], zero_division=0)), 3),
            "contact_rate": round(float(part["pred"].mean()), 3),
        }
        print(f"senior_citizen={g}: {slices[f'senior_citizen={g}']}")

    print("\n== Permutation importance (drop in ROC AUC, test set) ==")
    pi = permutation_importance(model, test[ALL_FEATURES], y_test, scoring="roc_auc",
                                n_repeats=10, random_state=42, n_jobs=1)
    imp = (pd.DataFrame({"feature": ALL_FEATURES,
                         "importance_mean": pi.importances_mean,
                         "importance_std": pi.importances_std})
           .sort_values("importance_mean", ascending=False))
    imp.round(4).to_csv("reports/permutation_importance.csv", index=False)
    print(imp.round(4).to_string(index=False))

    joblib.dump(model, "models/churn_model.joblib")
    reloaded = joblib.load("models/churn_model.joblib")
    assert np.allclose(reloaded.predict_proba(test[ALL_FEATURES])[:, 1], proba)

    metadata = {
        "model_name": "northstar-churn",
        "model_version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "HistGradientBoostingClassifier",
        "hyperparameters": tuning["best_params"],
        "decision_threshold": thr,
        "cost_assumptions": {"missed_churner": COST_FN, "contact": COST_CONTACT},
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "feature_order": ALL_FEATURES,
        "excluded_columns": LEAKY_FEATURES + [ID_COLUMN],
        "training_data_sha256": sha256_of("data/processed/train.csv"),
        "training_rows": int(len(train)),
        "test_metrics": metrics,
        "test_confusion": confusion,
        "subgroup_slices": slices,
        "reference_stats": {
            "mean_predicted_proba_test": round(float(proba.mean()), 4),
            "contact_rate_test": round(float(pred.mean()), 4),
        },
        "library_versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    top3 = ", ".join(imp["feature"].head(3))
    card = [
        f"# Model Card: northstar-churn v{MODEL_VERSION}",
        "",
        "## Intended use",
        "Monthly ranking of active NorthStar Telecom customers by churn risk so the retention",
        "team can decide whom to contact. Not for pricing, credit or any decision about",
        "individual eligibility for services.",
        "",
        "## Model",
        f"- Algorithm: HistGradientBoostingClassifier, params {tuning['best_params']}",
        f"- Decision threshold: {thr:.2f} (minimises cost with missed churner={COST_FN:.0f}, contact={COST_CONTACT:.0f})",
        f"- Features: {', '.join(ALL_FEATURES)}",
        f"- Excluded: {', '.join(LEAKY_FEATURES)} (post-outcome leakage), {ID_COLUMN}",
        "",
        "## Evaluation (hold-out test set, n=%d)" % len(test),
        f"- ROC AUC {metrics['roc_auc']}, PR AUC {metrics['pr_auc']}",
        f"- At threshold: precision {metrics['precision_at_threshold']}, recall {metrics['recall_at_threshold']}, contact rate {metrics['contact_rate_at_threshold']}",
        f"- Business cost at threshold {metrics['business_cost_at_threshold']:.0f} vs {metrics['business_cost_at_0.5']:.0f} at 0.5",
        f"- Most important features (permutation): {top3}",
        "",
        "## Subgroup check (senior_citizen)",
        *[f"- {k}: {v}" for k, v in slices.items()],
        "",
        "## Limitations",
        "- Trained on a synthetic extract; validate on real data before use.",
        "- Assumes every offer costs the same and that contact does not harm loyalty.",
        "- Performance will degrade under distribution drift; monitor PSI monthly.",
    ]
    with open("reports/model_card.md", "w") as f:
        f.write("\n".join(card) + "\n")
    print("\nsaved models/churn_model.joblib, models/metadata.json, reports/model_card.md")


if __name__ == "__main__":
    main()
