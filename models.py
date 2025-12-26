from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


# Feature names for augmented dataset
FEATURE_NAMES = [
    "GDP per capita, PPP (constant 2021 international $)",
    "Mean R&D Expenditure (% GDP)",
    "Mean Education Spending (% GDP)",
    "Academic Freedom Index",
    "Population",
    "Is Post-Soviet",
]

TARGET_NAME = "Researchers per Million"


@dataclass
class Metrics:
    r2: float
    mae: float
    mse: float
    n_samples: int = 0


def build_features_and_target(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build features and target from augmented dataset."""
    X = merged_df[FEATURE_NAMES]
    y = merged_df[TARGET_NAME]
    return X, y


def fit_full_model(merged_df: pd.DataFrame) -> Tuple[LinearRegression, Metrics]:
    """
    Fit a linear regression model on the full augmented dataset.
    """

    X, y = build_features_and_target(merged_df)
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = Metrics(
        r2=r2_score(y, y_pred),
        mae=mean_absolute_error(y, y_pred),
        mse=mean_squared_error(y, y_pred),
        n_samples=len(X),
    )

    return model, metrics


def train_test_split_data(
    merged_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split from augmented dataset.
    """

    X, y = build_features_and_target(merged_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def cross_validate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform K-fold cross-validation, returning average metrics
    (R², MAE, MSE) across folds.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores = []
    mae_scores = []
    mse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model_kf = LinearRegression()
        model_kf.fit(X_train_fold, y_train_fold)

        y_pred_kf = model_kf.predict(X_val_fold)

        r2_scores.append(r2_score(y_val_fold, y_pred_kf))
        mae_scores.append(mean_absolute_error(y_val_fold, y_pred_kf))
        mse_scores.append(mean_squared_error(y_val_fold, y_pred_kf))

    return {
        "avg_r2": float(np.mean(r2_scores)),
        "avg_mae": float(np.mean(mae_scores)),
        "avg_mse": float(np.mean(mse_scores)),
    }


def evaluate_on_test_set(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Metrics:
    """
    Train a final model on the training set and evaluate on the test set.
    """

    model_final = LinearRegression()
    model_final.fit(X_train, y_train)

    y_pred_test = model_final.predict(X_test)

    return Metrics(
        r2=r2_score(y_test, y_pred_test),
        mae=mean_absolute_error(y_test, y_pred_test),
        mse=mean_squared_error(y_test, y_pred_test),
        n_samples=len(X_test),
    )


