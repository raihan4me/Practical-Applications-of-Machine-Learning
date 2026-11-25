#!/usr/bin/env python3
"""Generate synthetic NorthStar Telecom customer data for the churn lab."""
import argparse

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "None"]
PAYMENTS = ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]


def generate(n, seed, drift=False):
    rng = np.random.default_rng(seed)
    contract_p = [0.70, 0.18, 0.12] if drift else [0.55, 0.25, 0.20]
    contract = rng.choice(CONTRACTS, size=n, p=contract_p)
    tenure = np.clip(np.round(rng.gamma(2.0, 14.0, size=n)), 1, 72).astype(int)
    internet = rng.choice(INTERNET, size=n, p=[0.35, 0.45, 0.20])
    base = np.select([internet == "Fiber optic", internet == "DSL"], [85.0, 55.0], default=25.0)
    monthly = base + rng.normal(0.0, 12.0, size=n) + (15.0 if drift else 0.0)
    monthly = np.round(np.clip(monthly, 18.0, 140.0), 2)
    total = np.round(tenure * monthly * rng.uniform(0.90, 1.05, size=n), 2)
    payment = rng.choice(PAYMENTS, size=n, p=[0.35, 0.22, 0.22, 0.21])
    support_calls = rng.poisson(2.5 if drift else 1.5, size=n)
    senior = rng.binomial(1, 0.16, size=n)
    num_products = rng.integers(1, 6, size=n)
    last_login = np.clip(np.round(rng.exponential(10.0, size=n)), 0, 120).astype(int)

    z = (-2.0
         + 1.3 * (contract == "Month-to-month")
         - 0.8 * (contract == "Two year")
         - 0.035 * tenure
         + 0.015 * (monthly - 60.0)
         + 0.35 * support_calls
         + 0.5 * (internet == "Fiber optic")
         + 0.8 * ((internet == "Fiber optic") & (monthly > 90.0))
         + 0.4 * (payment == "Electronic check")
         + 0.3 * senior
         + 0.02 * last_login
         - 0.2 * (num_products - 1)
         + rng.normal(0.0, 0.5, size=n))
    prob = 1.0 / (1.0 + np.exp(-z))
    churn = rng.binomial(1, prob)

    retention_call = np.where(churn == 1,
                              rng.binomial(1, 0.90, size=n),
                              rng.binomial(1, 0.03, size=n))

    df = pd.DataFrame({
        "customer_id": [f"NS{seed:03d}-{i:06d}" for i in range(n)],
        "tenure_months": tenure,
        "contract": contract,
        "internet_service": internet,
        "monthly_charges": monthly,
        "total_charges": total,
        "payment_method": payment,
        "support_calls": support_calls,
        "senior_citizen": senior,
        "num_products": num_products,
        "last_login_days": last_login,
        "retention_call_made": retention_call,
        "churn": churn,
    })
    df.loc[rng.random(n) < 0.02, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.03, "payment_method"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drift", action="store_true", help="simulate next-month distribution shift")
    ap.add_argument("--no-label", action="store_true", help="drop churn and post-outcome columns")
    ap.add_argument("--duplicates", type=int, default=0, help="number of duplicated rows to inject")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = generate(args.n, args.seed, args.drift)
    if args.duplicates:
        dups = df.sample(args.duplicates, random_state=args.seed)
        df = pd.concat([df, dups], ignore_index=True)
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.no_label:
        df = df.drop(columns=["churn", "retention_call_made"])
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows x {df.shape[1]} cols to {args.out}")


if __name__ == "__main__":
    main()
