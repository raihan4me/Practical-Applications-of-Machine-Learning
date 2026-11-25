# Practical Applications of Machine Learning

NorthStar Telecom churn prediction lab — end-to-end ML pipeline covering data quality,
leakage-safe preprocessing, model comparison, cost-optimal thresholding, packaging,
a FastAPI service, Docker deployment, and drift monitoring.

## Repository layout

.
├── src/
│ ├── generate_data.py # synthetic data generator
│ ├── audit.py # data quality + leakage screen
│ ├── features.py # shared preprocessing pipeline
│ ├── train_baseline.py # dummy + logreg + leakage demo
│ ├── compare_models.py # 5-fold CV model comparison
│ ├── tune.py # HGB tuning + cost-optimal threshold
│ ├── train_final.py # final model, explainability, packaging
│ └── monitor.py # PSI drift monitoring
├── serve/
│ ├── app.py # FastAPI service
│ ├── init.py
│ └── requirements.txt
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md

## Reproduce from scratch

```bash
# 1. Setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip==24.0
pip install -r requirements.txt

# 2. Project directories
mkdir -p data/raw data/processed data/incoming src serve models reports run

# 3. Generate data
python src/generate_data.py --n 8000 --seed 42 --duplicates 60 --out data/raw/customers.csv
python src/generate_data.py --n 2000 --seed 7 --drift --no-label --out data/incoming/batch_2024_07.csv

# 4. Data audit + leakage screen
python src/audit.py

# 5. Baselines
python src/train_baseline.py

# 6. Model comparison
python src/compare_models.py

# 7. Tuning + cost-optimal threshold
python src/tune.py

# 8. Final model + metadata + model card
python src/train_final.py

# 9. Serve locally
uvicorn serve.app:app --host 127.0.0.1 --port 8000

# 10. Containerize
docker build -t northstar/churn-api:1.0.0 .
docker run -d --name churn-api -p 8001:8000 --memory 1g northstar/churn-api:1.0.0

# 11. Batch scoring + drift
python src/monitor.py
Key results
Task	Result
Leakage suspect	retention_call_made (AUC ~0.93)
Honest logreg ROC AUC	~0.80
Leaky model ROC AUC	~0.97 (leak gap ~0.17)
Cost-optimal threshold	~0.10–0.15
Cost saving vs 0.50	~40–50%
Hold-out ROC AUC (gate ≥ 0.75)	~0.80 ✅
Drift recommendation	RETRAIN_RECOMMENDED
Major drift features	monthly_charges, support_calls
Library versions
Pinned in requirements.txt (and serve/requirements.txt):

text
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
scikit-learn==1.4.2
joblib==1.4.2
fastapi==0.110.3
uvicorn==0.29.0
pydantic==2.7.1
The pickled model requires scikit-learn 1.4.2 exactly. The Dockerfile installs the same pins so the container produces identical predictions to the local service.
