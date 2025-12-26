"""
Econometric Hypothesis Testing Module

This module implements essential econometric tests for linear regression analysis:
1. t-tests for individual coefficient significance (H0: β_i = 0)
2. F-test for overall model significance (H0: all β_i = 0)
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class CoefficientTest:
    """Results from t-test for a single coefficient."""
    feature_name: str
    coefficient: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float  # 95% confidence interval lower bound
    ci_upper: float  # 95% confidence interval upper bound
    is_significant_5pct: bool  # True if p < 0.05
    is_significant_1pct: bool  # True if p < 0.01


@dataclass
class FTestResult:
    """Results from F-test for overall model significance."""
    f_statistic: float
    p_value: float
    df_model: int  # degrees of freedom for model
    df_residual: int  # degrees of freedom for residuals
    is_significant: bool  # True if p < 0.05


@dataclass
class EconometricTestSuite:
    """Complete suite of econometric test results."""
    # Basic model info
    n_samples: int
    n_features: int
    r_squared: float
    adj_r_squared: float
    
    # Hypothesis tests
    coefficient_tests: List[CoefficientTest]
    f_test: FTestResult


def compute_coefficient_tests(
    X: pd.DataFrame,
    y: pd.Series,
    model: LinearRegression,
    feature_names: List[str]
) -> List[CoefficientTest]:
    """
    Compute t-tests for individual coefficient significance.
    
    H0: β_i = 0 (coefficient is zero, no effect)
    H1: β_i ≠ 0 (coefficient is non-zero, has effect)
    """
    n = len(X)
    k = X.shape[1]
    
    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Residual sum of squares
    rss = np.sum(residuals**2)
    
    # Residual standard error
    sigma_squared = rss / (n - k - 1)
    
    # Standard errors of coefficients
    # SE(β) = sqrt(σ² * (X'X)^(-1))
    X_with_intercept = np.column_stack([np.ones(n), X])
    try:
        cov_matrix = sigma_squared * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        cov_matrix = sigma_squared * np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
        std_errors = np.sqrt(np.diag(cov_matrix))
    
    # t-statistics and p-values
    coefficients = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefficients / std_errors
    
    # Two-tailed test
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))
    
    # 95% confidence intervals
    t_critical = stats.t.ppf(0.975, df=n - k - 1)
    ci_lower = coefficients - t_critical * std_errors
    ci_upper = coefficients + t_critical * std_errors
    
    # Create results (skip intercept, start from index 1)
    results = []
    for i, name in enumerate(feature_names):
        idx = i + 1  # Skip intercept
        results.append(CoefficientTest(
            feature_name=name,
            coefficient=coefficients[idx],
            std_error=std_errors[idx],
            t_statistic=t_stats[idx],
            p_value=p_values[idx],
            ci_lower=ci_lower[idx],
            ci_upper=ci_upper[idx],
            is_significant_5pct=p_values[idx] < 0.05,
            is_significant_1pct=p_values[idx] < 0.01,
        ))
    
    return results


def compute_f_test(
    X: pd.DataFrame,
    y: pd.Series,
    model: LinearRegression
) -> FTestResult:
    """
    Compute F-test for overall model significance.
    
    H0: β_1 = β_2 = ... = β_k = 0 (all coefficients are zero)
    H1: At least one β_i ≠ 0
    """
    n = len(X)
    k = X.shape[1]
    
    # Predictions and metrics
    y_pred = model.predict(X)
    
    # Total sum of squares
    tss = np.sum((y - np.mean(y))**2)
    
    # Residual sum of squares
    rss = np.sum((y - y_pred)**2)
    
    # Explained sum of squares
    ess = tss - rss
    
    # F-statistic
    f_stat = (ess / k) / (rss / (n - k - 1))
    
    # p-value
    p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    
    return FTestResult(
        f_statistic=f_stat,
        p_value=p_value,
        df_model=k,
        df_residual=n - k - 1,
        is_significant=p_value < 0.05
    )


def run_econometric_tests(
    X: pd.DataFrame,
    y: pd.Series,
    model: LinearRegression,
    feature_names: List[str]
) -> EconometricTestSuite:
    """
    Run econometric hypothesis tests.
    
    Args:
        X: Feature matrix
        y: Target variable
        model: Fitted linear regression model
        feature_names: Names of features
    
    Returns:
        EconometricTestSuite with all test results
    """
    n = len(X)
    k = X.shape[1]
    
    # R-squared and adjusted R-squared
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    # Run tests
    coefficient_tests = compute_coefficient_tests(X, y, model, feature_names)
    f_test = compute_f_test(X, y, model)
    
    return EconometricTestSuite(
        n_samples=n,
        n_features=k,
        r_squared=r2,
        adj_r_squared=adj_r2,
        coefficient_tests=coefficient_tests,
        f_test=f_test
    )


def print_test_results(results: EconometricTestSuite) -> None:
    """Print formatted econometric test results."""
    
    print("\n" + "=" * 100)
    print("ECONOMETRIC HYPOTHESIS TESTING RESULTS")
    print("=" * 100)
    
    # Basic model info
    print(f"\n📊 MODEL SUMMARY")
    print(f"   Observations: {results.n_samples}")
    print(f"   Features: {results.n_features}")
    print(f"   R²: {results.r_squared:.4f}")
    print(f"   Adjusted R²: {results.adj_r_squared:.4f}")
    
    # 1. Coefficient significance (t-tests)
    print(f"\n{'=' * 100}")
    print("1. COEFFICIENT SIGNIFICANCE TESTS (t-tests)")
    print(f"{'=' * 100}")
    print(f"   H₀: β_i = 0 (coefficient has no effect)")
    print(f"   H₁: β_i ≠ 0 (coefficient has significant effect)\n")
    
    print(f"{'Feature':<50} {'Coeff':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10} {'Sig.':<10}")
    print("-" * 100)
    for test in results.coefficient_tests:
        sig_stars = ""
        if test.is_significant_1pct:
            sig_stars = "***"
        elif test.is_significant_5pct:
            sig_stars = "**"
        
        print(f"{test.feature_name:<50} {test.coefficient:>10.4f} {test.std_error:>10.4f} "
              f"{test.t_statistic:>10.4f} {test.p_value:>10.4f} {sig_stars:<10}")
    
    print("\n   Significance levels: *** p<0.01, ** p<0.05")
    
    # Show confidence intervals
    print(f"\n   95% Confidence Intervals:")
    for test in results.coefficient_tests:
        print(f"   {test.feature_name:<50} [{test.ci_lower:>10.4f}, {test.ci_upper:>10.4f}]")
    
    # 2. F-test for overall significance
    print(f"\n{'=' * 100}")
    print("2. OVERALL MODEL SIGNIFICANCE (F-test)")
    print(f"{'=' * 100}")
    print(f"   H₀: All coefficients are zero (model has no explanatory power)")
    print(f"   H₁: At least one coefficient is non-zero\n")
    print(f"   F-statistic: {results.f_test.f_statistic:.4f}")
    print(f"   p-value: {results.f_test.p_value:.6f}")
    print(f"   df: ({results.f_test.df_model}, {results.f_test.df_residual})")
    
    if results.f_test.is_significant:
        print(f"   ✅ RESULT: Reject H₀ - Model is statistically significant (p < 0.05)")
    else:
        print(f"   ❌ RESULT: Fail to reject H₀ - Model lacks significance (p ≥ 0.05)")
    
    print(f"\n{'=' * 100}\n")
