#!/usr/bin/env python3
"""Task 8: score an incoming batch and check data/prediction drift with PSI."""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from features import ALL_FEATURES, CATEGORICAL_FEATURES

PSI_MODERATE = 0.10
PSI_MAJOR = 0.20
PRED_SHIFT_LIMIT = 0.05
BATCH = "data/incoming/batch_2024_07.csv"


def psi_categorical(ref, cur):
    ref = pd.Series(ref).astype(object).where(pd.Series(ref).notna(), "__missing__").astype(str)
    cur = pd.Series(cur).astype(object).where(pd.Series(cur).notna(), "__missing__").astype(str)
    cats = sorted(set(ref) | set(cur))
    r = np.clip(ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    c = np.clip(cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, cur, bins=10):
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, bins=edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_psi(ref, cur):
    if ref.name in CATEGORICAL_FEATURES or ref.dropna().nunique() <= 10:
        return psi_categorical(ref, cur), "categorical"
    return psi_numeric(ref, cur), "binned"


def main():
    meta = json.load(open("models/metadata.json"))
    model = joblib.load("models/churn_model.joblib")
    ref = pd.read_csv("data/processed/train.csv", keep_default_na=False, na_values=[""])
    batch = pd.read_csv(BATCH)

    missing = set(ALL_FEATURES) - set(batch.columns)
    if missing:
        sys.exit(f"schema check failed, batch is missing columns: {sorted(missing)}")

    print(f"Feature drift (reference=training data n={len(ref)}, current={BATCH} n={len(batch)})")
    features = {}
    for col in ALL_FEATURES:
        psi, method = feature_psi(ref[col], batch[col])
        status = "MAJOR" if psi >= PSI_MAJOR else "MODERATE" if psi >= PSI_MODERATE else "OK"
        features[col] = {"psi": round(psi, 4), "method": method, "status": status}
        print(f"  {col:18s} psi={psi:6.3f} [{method:11s}] {status}")

    proba = model.predict_proba(batch[ALL_FEATURES])[:, 1]
    thr = float(meta["decision_threshold"])
    ref_mean = float(meta["reference_stats"]["mean_predicted_proba_test"])
    batch_mean = float(proba.mean())
    contact_rate = float((proba >= thr).mean())
    print("\nPrediction drift")
    print(f"  mean churn probability: reference={ref_mean:.3f} batch={batch_mean:.3f} "
          f"(delta={batch_mean - ref_mean:+.3f})")
    print(f"  contact rate at threshold {thr:.2f}: reference="
          f"{meta['reference_stats']['contact_rate_test']:.3f} batch={contact_rate:.3f}")

    scored = batch[["customer_id"]].copy()
    scored["churn_probability"] = proba.round(4)
    scored["contact_customer"] = proba >= thr
    scored.to_csv("data/incoming/batch_2024_07_scored.csv", index=False)

    major = [c for c, v in features.items() if v["status"] == "MAJOR"]
    retrain = bool(major) or abs(batch_mean - ref_mean) > PRED_SHIFT_LIMIT
    report = {
        "model_version": meta["model_version"],
        "batch": BATCH,
        "rows": int(len(batch)),
        "features": features,
        "prediction": {"reference_mean": ref_mean, "batch_mean": round(batch_mean, 4),
                       "batch_contact_rate": round(contact_rate, 4)},
        "major_drift_features": major,
        "recommendation": "RETRAIN_RECOMMENDED" if retrain else "NO_ACTION",
    }
    with open("reports/drift_report.json", "w") as f:
        json.dump(report, f, indent=2)

    sample = json.loads(batch.head(3).to_json(orient="records"))
    with open("run/batch_sample_request.json", "w") as f:
        json.dump({"customers": sample}, f, indent=2)

    print(f"\nrecommendation: {report['recommendation']} (major drift: {major})")
    print("saved reports/drift_report.json and data/incoming/batch_2024_07_scored.csv")


if __name__ == "__main__":
    main()
