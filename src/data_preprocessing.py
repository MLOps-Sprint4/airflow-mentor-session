"""
Data loading and preprocessing utilities for the Telco Churn Airflow DAGs.

Imported inside DAG task bodies (not at DAG parse time).
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "telco_churn_full.csv")

# Columns to drop (not useful for modelling)
DROP_COLS = ["customerID", "RecordDate"]

# Target column
TARGET = "Churn"

# Columns that are already numeric in the raw data
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen is coded as 0/1 in the CSV - treat it as numeric
PASSTHROUGH_NUMERIC = ["SeniorCitizen"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the Telco Churn CSV and perform minimal cleaning.

    Steps:
        1. Read the CSV.
        2. Drop identifier / date columns.
        3. Convert ``TotalCharges`` to numeric (blanks become NaN, then
           filled with 0 - these correspond to brand-new customers with
           tenure == 0).
        4. Map the target ``Churn`` to binary (Yes=1, No=0).

    Parameters
    ----------
    path : str or None
        Path to the CSV file.  Defaults to ``data/telco_churn_full.csv``
        relative to the project root.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for preprocessing.
    """
    if path is None:
        path = DEFAULT_DATA_PATH

    df = pd.read_csv(path)

    # Drop columns that aren't features
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # TotalCharges may contain spaces for new customers - coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler, list[str]]:
    """Encode categoricals and scale numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_data`.
    scaler : StandardScaler or None
        If provided and ``fit_scaler`` is False, uses this pre-fitted scaler.
    fit_scaler : bool
        Whether to fit the scaler on the data (True for training, False for
        inference).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all numeric).
    y : pd.Series
        Binary target.
    scaler : StandardScaler
        Fitted scaler (useful for saving / reuse).
    feature_names : list[str]
        Column names of X, in order.
    """
    df = df.copy()
    y = df.pop(TARGET)

    # --- Label-encode binary categorical columns ---
    binary_cols = []
    for col in df.select_dtypes(include="object").columns:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            binary_cols.append(col)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # --- One-hot encode remaining categorical columns ---
    remaining_cat = [
        c
        for c in df.select_dtypes(include="object").columns
        if c not in binary_cols
    ]
    if remaining_cat:
        df = pd.get_dummies(df, columns=remaining_cat, drop_first=True)

    # Ensure all columns are numeric (safety net)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # --- Scale numeric features ---
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    feature_names = list(df.columns)

    return df, y, scaler, feature_names


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
