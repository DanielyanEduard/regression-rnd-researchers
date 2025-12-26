from data_loader import load_all_datasets
from data_prep import build_augmented_dataset
from eda_augmented import run_augmented_eda
from models import (
    fit_full_model,
    train_test_split_data,
    evaluate_on_test_set,
    FEATURE_NAMES,
)


def main(run_eda: bool = True, min_year: int = 2019, max_year: int = 2023) -> None:
    """
    Main training function using augmented dataset approach.
    
    Instead of training separate models for each year, this builds one model
    using all available data across years. Each country-year combination becomes
    a separate training example.
    
    Args:
        run_eda: Whether to run EDA visualizations
        min_year: Minimum year to include in dataset (default: 2019)
        max_year: Maximum year to include in dataset (default: 2023)
    """
    
    # Load raw datasets
    print("Loading datasets...")
    (researcher_df, spending_df, education_df, 
     academic_freedom_df, population_df) = load_all_datasets()
    print("Datasets loaded successfully!")
    
    # Build augmented dataset
    print(f"\n{'='*80}")
    print(f"BUILDING AUGMENTED DATASET ({min_year}-{max_year})")
    print(f"{'='*80}")
    print(f"Each country can contribute multiple observations across years.")
    print(f"For each year X, features are computed relative to X:")
    print(f"  - GDP per capita in year X")
    print(f"  - Mean R&D spending in (X-4) to X")
    print(f"  - Mean education spending in (X-4) to X")
    print(f"  - Academic freedom in year X")
    print(f"  - Population in year X")
    print(f"  - Post-Soviet flag (constant)\n")
    
    merged_df = build_augmented_dataset(
        researcher_df, spending_df, education_df,
        academic_freedom_df, population_df,
        min_year=min_year, max_year=max_year
    )
    
    if len(merged_df) == 0:
        print("❌ ERROR: No data available for the specified year range.")
        return
    
    print(f"\n📊 Augmented dataset shape: {merged_df.shape}")
    print(f"   Total observations: {len(merged_df)}")
    print(f"   Unique countries: {merged_df['Entity'].nunique()}")
    print(f"   Years covered: {sorted(merged_df['Year'].unique())}")
    
    # Show data distribution by year
    year_counts = merged_df['Year'].value_counts().sort_index()
    print(f"\n   Observations per year:")
    for year, count in year_counts.items():
        print(f"     {year}: {count} observations")
    
    # Check if we have enough data
    if len(merged_df) < 20:
        print(f"\n⚠️  WARNING: Very limited data ({len(merged_df)} observations)")
        print(f"   Model may not be reliable. Consider widening year range.")
    
    # Optional EDA for augmented dataset
    if run_eda:
        print(f"\n{'='*80}")
        print("RUNNING AUGMENTED EDA")
        print(f"{'='*80}")
        run_augmented_eda(merged_df, save=True)
    
    # Train model on augmented dataset
    print(f"\n{'='*80}")
    print("TRAINING AUGMENTED MODEL")
    print(f"{'='*80}")
    
    # Fit model on full dataset
    full_model, full_metrics = fit_full_model(merged_df)
    
    print(f"\nFull-data model (in-sample):")
    print(f"  Observations: {full_metrics.n_samples}")
    print(f"  Intercept: {full_model.intercept_:.4f}")
    print(f"  Coefficients:")
    for name, coef in zip(FEATURE_NAMES, full_model.coef_):
        print(f"    {name}: {coef:.4f}")
    print(f"  R-squared: {full_metrics.r2:.4f}")
    print(f"  MAE: {full_metrics.mae:.2f}")
    print(f"  MSE: {full_metrics.mse:.2f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(merged_df)
    print(
        f"\nTrain/test split (80/20):"
        f"\n  X_train shape: {X_train.shape}"
        f"\n  X_test shape:  {X_test.shape}"
        f"\n  y_train shape: {y_train.shape}"
        f"\n  y_test shape:  {y_test.shape}"
    )
    
    # Final evaluation on test set
    test_metrics = evaluate_on_test_set(X_train, X_test, y_train, y_test)
    print(f"\nModel performance on held-out test set:")
    print(f"  Test observations: {test_metrics.n_samples}")
    print(f"  R-squared: {test_metrics.r2:.4f}")
    print(f"  MAE: {test_metrics.mae:.2f}")
    print(f"  MSE: {test_metrics.mse:.2f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully trained model on augmented dataset")
    print(f"   • Total observations: {len(merged_df)}")
    print(f"   • Unique countries: {merged_df['Entity'].nunique()}")
    print(f"   • Year range: {min_year}-{max_year}")
    print(f"   • Test R²: {test_metrics.r2:.4f}")
    print(f"   • Test MAE: {test_metrics.mae:.2f} researchers/million")
    print(f"\n💡 This model uses data augmentation: each country contributes")
    print(f"   multiple training examples across different years, increasing")
    print(f"   the effective dataset size and model robustness.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


