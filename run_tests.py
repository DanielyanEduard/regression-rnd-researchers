"""
Quick script to run econometric hypothesis tests on the trained regression model.
This skips EDA for faster execution.
"""

from train_regression import main

if __name__ == "__main__":
    # Run without EDA, only with econometric tests
    main(run_eda=False, run_tests=True, min_year=2019, max_year=2023)

