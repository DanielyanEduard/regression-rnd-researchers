from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


DEPENDENT_VARIABLE = "Researchers in R&D (per million people) in 2022"
INDEPENDENT_VARIABLES = [
    "Mean Researchers R&D (2017-2021)",
    "GDP per capita, PPP (constant 2021 international $)",
    "Mean Research and development expenditure (% of GDP) (2018-2022)",
    "Mean public spending on education as a share of GDP (historical and recent) (2018-2022)",
]


@dataclass
class Metrics:
    r2: float
    mae: float
    mse: float


def build_features_and_target(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = merged_df[INDEPENDENT_VARIABLES]
    y = merged_df[DEPENDENT_VARIABLE]
    return X, y


def fit_full_model(merged_df: pd.DataFrame) -> Tuple[LinearRegression, Metrics]:
    """
    Fit a linear regression model on the full dataset,
    mirroring the initial notebook fit.
    """

    X, y = build_features_and_target(merged_df)
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = Metrics(
        r2=r2_score(y, y_pred),
        mae=mean_absolute_error(y, y_pred),
        mse=mean_squared_error(y, y_pred),
    )

    return model, metrics


def train_test_split_data(
    merged_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split as in the notebook.
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
    Train a final model on the training set and evaluate on the test set,
    mirroring the notebook's held-out evaluation.
    """

    model_final = LinearRegression()
    model_final.fit(X_train, y_train)

    y_pred_test = model_final.predict(X_test)

    return Metrics(
        r2=r2_score(y_test, y_pred_test),
        mae=mean_absolute_error(y_test, y_pred_test),
        mse=mean_squared_error(y_test, y_pred_test),
    )


