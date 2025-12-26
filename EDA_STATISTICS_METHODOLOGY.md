# EDA Statistics Calculation Methodology (2019-2023)

## Overview

All EDA visualizations are based on the augmented dataset with **301 observations** from **87 countries** across **2019-2023**. Statistics are calculated differently based on whether the feature represents a **current year value** or a **past 5-year average**.

## Statistics by Feature Type

### 1. Current Year Features (Direct 2019-2023 Data)

These features represent values from the specific year of each observation:

#### **Researchers per Million**
- **Calculation**: Mean and median of all 301 observations from 2019-2023
- **Legend**: "Mean (2019-2023): 2685" | "Median (2019-2023): 1849"
- **Interpretation**: Average researcher density across all country-year combinations

#### **GDP per Capita**
- **Calculation**: Mean and median of GDP values from 2019-2023 observations
- **Legend**: "Mean (2019-2023): $38,246" | "Median (2019-2023): $34,703"
- **X-axis label**: "GDP per capita (PPP, constant 2021 $, current year)"
- **Interpretation**: Average wealth across all country-year combinations

#### **Academic Freedom Index**
- **Calculation**: Mean and median of academic freedom values from 2019-2023
- **Legend**: "Mean (2019-2023): 0.692" | "Median (2019-2023): 0.842"
- **X-axis label**: "Academic Freedom Index (current year)"
- **Interpretation**: Average academic freedom across all country-year combinations

#### **Population**
- **Calculation**: Mean and median of population values from 2019-2023
- **Interpretation**: Population size for each country-year observation

### 2. Past 5-Year Average Features (Pre-Averaged Data)

These features are **already averaged** when the dataset is built. Each observation contains a mean calculated over the previous 5 years relative to that observation's target year.

#### **Mean R&D Expenditure (% GDP)**
- **Feature Construction**: For each observation in year X, this is the mean of R&D spending from (X-4) to X
  - Example: For 2020 observation → mean of 2016-2020
  - Example: For 2023 observation → mean of 2019-2023
- **Statistics Calculation**: Mean and median of these pre-calculated means across all 301 observations
- **Legend**: "Mean (2019-2023): 1.10%" | "Median (2019-2023): 0.76%"
- **X-axis label**: "Mean R&D Expenditure (past 5 years per obs, % GDP)"
- **Interpretation**: Average of the 5-year rolling means across all observations

#### **Mean Education Spending (% GDP)**
- **Feature Construction**: For each observation in year X, this is the mean of education spending from (X-4) to X
  - Example: For 2020 observation → mean of 2016-2020
  - Example: For 2023 observation → mean of 2019-2023
- **Statistics Calculation**: Mean and median of these pre-calculated means across all 301 observations
- **Legend**: "Mean (2019-2023): 4.52%" | "Median (2019-2023): 4.44%"
- **X-axis label**: "Mean Education Spending (past 5 years per obs, % GDP)"
- **Interpretation**: Average of the 5-year rolling means across all observations

## Detailed Example: R&D Expenditure

Let's trace how R&D expenditure statistics are calculated:

### Step 1: Feature Preparation (in `data_prep.py`)
For a country with data in 2020, 2021, 2022:

| Year (X) | R&D Values Used | Mean R&D (feature value) |
|----------|----------------|-------------------------|
| 2020 | 2016, 2017, 2018, 2019, 2020 | e.g., 1.2% |
| 2021 | 2017, 2018, 2019, 2020, 2021 | e.g., 1.3% |
| 2022 | 2018, 2019, 2020, 2021, 2022 | e.g., 1.4% |

### Step 2: EDA Statistics (in `eda_augmented.py`)
The mean shown in the legend is:
```python
rd_mean = merged_df['Mean R&D Expenditure (% GDP)'].mean()
# This calculates: mean of [1.2%, 1.3%, 1.4%, ..., for all 301 obs]
```

### Step 3: Interpretation
- **"Mean (2019-2023): 1.10%"** means:
  - Across all 301 observations (2019-2023)
  - The average of their 5-year rolling means
  - Is 1.10%

This is NOT the same as taking the mean of all R&D spending values from 2015-2023. Instead, it's the mean of pre-calculated 5-year averages.

## Why This Matters

### Current Year Features
- **Direct measurement**: We see the actual distribution of values in 2019-2023
- **Simple interpretation**: "The average GDP per capita in our dataset is $38,246"

### Past 5-Year Features
- **Smoothed measurement**: Each observation already has built-in temporal smoothing
- **Layered interpretation**: "The average of 5-year rolling means is 1.10%"
- **Reduces noise**: By averaging over 5 years first, we reduce year-to-year volatility

## Summary Table

| Feature | Data Type | Statistic Calculation | Time Window |
|---------|-----------|----------------------|-------------|
| Researchers per Million | Current year | Mean/Median of 301 obs | 2019-2023 |
| GDP per Capita | Current year | Mean/Median of 301 obs | 2019-2023 |
| Academic Freedom | Current year | Mean/Median of 301 obs | 2019-2023 |
| Population | Current year | Mean/Median of 301 obs | 2019-2023 |
| R&D Expenditure | 5-year rolling avg | Mean/Median of pre-averaged values | (X-4) to X per obs |
| Education Spending | 5-year rolling avg | Mean/Median of pre-averaged values | (X-4) to X per obs |

## Label Conventions

### Legend Labels
- **Current year features**: "Mean (2019-2023): value"
- **Past 5-year features**: "Mean (2019-2023): value" (mean of rolling averages)

### X-Axis Labels
- **Current year features**: "Feature Name (current year)"
- **Past 5-year features**: "Mean Feature Name (past 5 years per obs, unit)"

The "(past 5 years per obs)" clarifies that each observation point on the histogram represents a 5-year average specific to that observation's year.

## Code Implementation

### Current Year Example (GDP):
```python
# Direct calculation
gdp_mean = merged_df['GDP per capita, PPP (constant 2021 international $)'].mean()
# Result: mean of all 301 GDP values from 2019-2023
```

### Past 5-Year Example (R&D):
```python
# Calculation on pre-averaged data
rd_mean = merged_df['Mean R&D Expenditure (% GDP)'].mean()
# Result: mean of 301 pre-calculated 5-year averages
# Each of the 301 values is itself a mean of 5 years
```

## Verification

To verify statistics are correct:
1. **Current year features**: Check that mean/median match dataset summary
2. **Past 5-year features**: Verify each observation has correct rolling average
3. **All features**: Ensure labels clearly indicate calculation methodology

---

**Updated**: 2025-12-25
**Dataset**: 301 observations from 87 countries (2019-2023)
**Methodology**: Augmented dataset with dynamic feature calculation

