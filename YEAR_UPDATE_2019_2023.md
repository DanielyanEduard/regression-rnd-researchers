# Year Range Update: 2019-2023

## Summary

The project's target year range has been updated from **2020-2024** to **2019-2023** throughout all code and documentation.

## Changes Made

### 1. Code Files Updated

#### `train_regression.py`
- Default `min_year` parameter: `2020` → `2019`
- Default `max_year` parameter: `2024` → `2023`
- Function documentation updated

#### `data_prep.py`
- `build_augmented_dataset()` default `min_year`: `2020` → `2019`
- `build_augmented_dataset()` default `max_year`: `2024` → `2023`
- Function documentation updated

### 2. Documentation Files Updated

#### `GLOBAL_VISUALIZATIONS.md`
- Dataset observations: 228 → 301
- Unique countries: 84 → 87
- Time period: 2020-2023 → 2019-2023
- Model metrics updated to reflect new data
- Year distribution updated to include 2019

#### `README.md`
- Project title updated to reflect augmented dataset approach
- Year range updated throughout
- Pipeline overview updated
- What's New section updated

### 3. Impact on Model Performance

#### Before (2020-2024):
- **Observations**: 228
- **Countries**: 84
- **Test R²**: 0.8656
- **Test MAE**: 682.70 researchers/million

#### After (2019-2023):
- **Observations**: 301 (+32%)
- **Countries**: 87 (+4%)
- **Test R²**: 0.8760 (+1.2%)
- **Test MAE**: 576.86 (-15.5% improvement)

### 4. Benefits of Including 2019

1. **More Training Data**: 73 additional observations from 2019
2. **Better Coverage**: 3 additional countries
3. **Improved Generalization**: Model performance improved on test set
4. **Better MAE**: Lower mean absolute error (576.86 vs 682.70)
5. **More Robust**: Larger training dataset reduces overfitting risk

### 5. Year Distribution

```
2019: 73 observations
2020: 69 observations
2021: 72 observations
2022: 65 observations
2023: 22 observations
Total: 301 observations
```

Note: 2023 has fewer observations due to data availability lag (countries report data with delay).

## Visualization Updates

All global visualizations now reflect the 2019-2023 time period:

1. **`global_distributions.png`**: Updated summary table showing 2019-2023
2. **`global_relationships.png`**: Shows patterns across 301 observations
3. **`global_correlations.png`**: Correlation analysis with full dataset
4. **`global_top_countries.png`**: Country rankings based on 2019-2023 data

## Technical Details

### Feature Calculation
For each year X in the range 2019-2023:
- **GDP per capita**: Value in year X
- **R&D Expenditure**: Mean of (X-4) to X (e.g., 2015-2019 for year 2019)
- **Education Spending**: Mean of (X-4) to X
- **Academic Freedom**: Value in year X
- **Population**: Value in year X
- **Post-Soviet Flag**: Constant (0 or 1)

### Why 2019-2023?
- **2019 inclusion**: Adds significant data (73 observations) with complete features
- **2023 exclusion of 2024**: 2024 data not yet available or too sparse
- **Balance**: Good balance between data volume and data quality
- **Recent data**: Focuses on most recent 5-year period

## Model Coefficients Comparison

| Feature | 2020-2024 | 2019-2023 | Change |
|---------|-----------|-----------|--------|
| Intercept | -507.93 | -482.65 | +25.28 |
| GDP per capita | 0.0273 | 0.0275 | +0.0002 |
| R&D Expenditure | 1867.43 | 1828.84 | -38.59 |
| Education Spending | -14.95 | -11.72 | +3.23 |
| Academic Freedom | 269.82 | 250.57 | -19.25 |
| Population | -0.0000 | -0.0000 | 0.0000 |
| Post-Soviet | 548.44 | 526.06 | -22.38 |

**Interpretation**: Coefficients are relatively stable, indicating the model is robust to the year range change. The improvements in R² and MAE come from having more training data rather than fundamentally different relationships.

## Recommendations

1. **Monitor 2024 data**: When 2024 data becomes available, consider updating to 2020-2024
2. **Keep augmented approach**: The data augmentation strategy proves valuable
3. **Maintain 5-year window**: The 5-year period balances recency with data volume
4. **Track new countries**: The additional 3 countries (84→87) suggest ongoing data expansion

---

**Updated**: 2025-12-25
**Year Range**: 2019-2023
**Total Observations**: 301 from 87 countries
**Model Performance**: Test R² = 0.8760, MAE = 576.86

