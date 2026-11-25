#!/usr/bin/env bash
# Final validation for the U006 Practical Applications of Machine Learning lab
cd "$HOME/ml-lab" || exit 1
source .venv/bin/activate
PASS=0; FAIL=0
check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "[PASS] $desc"; PASS=$((PASS+1))
  else echo "[FAIL] $desc"; FAIL=$((FAIL+1)); fi
}

check "Setup: scikit-learn 1.4.2 pinned in venv" \
  python -c "import sklearn; assert sklearn.__version__ == '1.4.2'"
check "Task 1: raw extract has 8060 data rows" \
  bash -c '[ "$(($(wc -l < data/raw/customers.csv) - 1))" -eq 8060 ]'
check "Task 1: clean data has 8000 unique rows" \
  python -c "import pandas as pd; d=pd.read_csv('data/processed/customers_clean.csv'); assert len(d)==8000 and d.duplicated().sum()==0"
check "Task 2: preprocessor yields 17 dense features on train" \
  env PYTHONPATH=src python -c "import pandas as pd; from features import build_preprocessor, ALL_FEATURES; d=pd.read_csv('data/processed/train.csv'); assert build_preprocessor().fit_transform(d[ALL_FEATURES]).shape==(6400,17)"
check "Task 2: leakage demonstrated (leaky AUC > 0.95 > honest AUC > 0.70)" \
  python -c "import json; r={x['model']:x for x in json.load(open('reports/baseline.json'))}; assert r['logreg_WITH_LEAK']['roc_auc']>0.95>r['logreg']['roc_auc']>0.70"
check "Task 3: five models compared incl. knn" \
  python -c "import pandas as pd; d=pd.read_csv('reports/model_comparison.csv'); assert len(d)==5 and 'knn' in set(d.model)"
check "Task 4: cost-optimal threshold < 0.5 and beats naive policies" \
  python -c "import json; t=json.load(open('reports/tuning.json')); assert 0<t['threshold']<0.5 and t['oof_cost_at_threshold']<min(t['oof_cost_at_0.5'], t['cost_contact_all'], t['cost_contact_none'])"
check "Task 5: model artifact, metadata and model card exist" \
  test -f models/churn_model.joblib -a -f models/metadata.json -a -f reports/model_card.md
check "Task 5: hold-out ROC AUC passes quality gate, leak column excluded" \
  python -c "import json; m=json.load(open('models/metadata.json')); assert m['test_metrics']['roc_auc']>=0.75 and 'retention_call_made' not in m['feature_order']"
check "Task 6: schema restricts contract with Literal" \
  grep -Eq "contract: *Literal" serve/app.py
check "Task 7: container churn-api is healthy" \
  bash -c '[ "$(docker inspect -f "{{.State.Health.Status}}" churn-api)" = "healthy" ]'
check "Task 7: container runs as non-root" \
  bash -c '[ "$(docker exec churn-api id -u)" != "0" ]'
check "Task 7: /predict returns 2 predictions" \
  bash -c 'curl -sf -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d @run/sample_payload.json | jq -e ".predictions | length == 2"'
check "Task 7: invalid categories rejected with HTTP 422" \
  bash -c '[ "$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d @run/invalid_payload.json)" = "422" ]'
check "Task 8: major drift on monthly_charges and support_calls" \
  jq -e '.features.monthly_charges.status == "MAJOR" and .features.support_calls.status == "MAJOR"' reports/drift_report.json
check "Task 8: retraining recommended" \
  jq -e '.recommendation == "RETRAIN_RECOMMENDED"' reports/drift_report.json
check "Task 8: scored batch has 2000 rows" \
  bash -c '[ "$(($(wc -l < data/incoming/batch_2024_07_scored.csv) - 1))" -eq 2000 ]'

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
