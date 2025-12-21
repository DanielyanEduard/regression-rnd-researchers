from data_loader import load_all_datasets
from data_prep import build_merged_dataset
from eda import run_basic_eda
from models import (
    fit_full_model,
    train_test_split_data,
    cross_validate_model,
    evaluate_on_test_set,
)


def main(run_eda: bool = True) -> None:
    # 1. Load raw datasets
    researcher_df, spending_df, education_df = load_all_datasets()

    # 2. Build merged modeling dataset
    merged_df = build_merged_dataset(researcher_df, spending_df, education_df)
    print(f"Merged dataset shape: {merged_df.shape}")

    # 3. Optional EDA
    if run_eda:
        print("Running EDA and saving figures to the figures/ directory...")
        run_basic_eda(researcher_df, spending_df, education_df, merged_df, save=True)

    # 4. Fit model on full dataset (for intercept, coefficients, in-sample R²)
    full_model, full_metrics = fit_full_model(merged_df)
    print("\nFull-data model (in-sample):")
    print("  Intercept:", full_model.intercept_)
    print("  Coefficients:", full_model.coef_)
    print(f"  R-squared: {full_metrics.r2:.4f}")
    print(f"  MAE: {full_metrics.mae:.2f}")
    print(f"  MSE: {full_metrics.mse:.2f}")

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(merged_df)
    print(
        f"\nTrain/test split shapes: "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    # 6. K-fold cross-validation on the training set
    cv_results = cross_validate_model(X_train, y_train)
    print("\nK-fold cross-validation (training set):")
    print(f"  Average R-squared: {cv_results['avg_r2']:.4f}")
    print(f"  Average MAE: {cv_results['avg_mae']:.2f}")
    print(f"  Average MSE: {cv_results['avg_mse']:.2f}")

    # 7. Final evaluation on held-out test set
    test_metrics = evaluate_on_test_set(X_train, X_test, y_train, y_test)
    print("\nModel performance on held-out test set:")
    print(f"  R-squared: {test_metrics.r2:.4f}")
    print(f"  MAE: {test_metrics.mae:.2f}")
    print(f"  MSE: {test_metrics.mse:.2f}")


if __name__ == "__main__":
    main()


