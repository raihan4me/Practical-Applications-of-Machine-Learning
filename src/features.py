"""Shared feature definitions and preprocessing for the NorthStar churn model."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "customer_id"
TARGET = "churn"
LEAKY_FEATURES = ["retention_call_made"]

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges", "support_calls",
    "senior_citizen", "num_products", "last_login_days",
]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class NonePreservingImputer(BaseEstimator, TransformerMixin):
    """
    Fill missing values with the most frequent category.

    For the 'internet_service' column, a missing/NaN value represents the
    literal category "None" (no internet service). pandas' default read_csv
    converts the string "None" to NaN, so we restore it here to keep the
    encoder vocabulary at 3 categories (DSL, Fiber optic, None).
    """

    def __init__(self, none_category="None", none_column="internet_service"):
        self.none_category = none_category
        self.none_column = none_column

    def fit(self, X, y=None):
        # Restore the literal "None" category before computing the most frequent
        # fallback for other columns.
        X = self._restore(X)
        df = self._as_frame(X)
        self.fill_values_ = {}
        for col in df.columns:
            mode = df[col].dropna()
            self.fill_values_[col] = mode.mode().iloc[0] if len(mode) else "missing"
        return self

    def transform(self, X):
        X = self._restore(X)
        df = self._as_frame(X).copy()
        for col in df.columns:
            df[col] = df[col].fillna(self.fill_values_.get(col, "missing"))
        return df.to_numpy(dtype=object)

    # ---- helpers ----
    def _restore(self, X):
        df = self._as_frame(X).copy()
        if self.none_column in df.columns:
            df[self.none_column] = df[self.none_column].where(
                df[self.none_column].notna(), self.none_category
            )
        return df

    @staticmethod
    def _as_frame(X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def build_preprocessor(numeric=None, categorical=None):
    numeric = list(numeric or NUMERIC_FEATURES)
    categorical = list(categorical or CATEGORICAL_FEATURES)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", NonePreservingImputer()),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
    return preprocessor
