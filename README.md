# R&D Researchers - Econometric Analysis

This project implements a comprehensive econometric analysis of the relationship between research capacity (researchers per million people) and key economic, institutional, and policy factors using data from **[Our World in Data](https://ourworldindata.org/)**.

## Overview

The analysis uses an augmented panel dataset approach where each country contributes multiple observations across years (2019-2023), building a robust linear regression model with 301 observations from 87 countries.

### Dependent Variable
- **Researchers in R&D per million people**

### Independent Variables (6 predictors)
1. **GDP per capita** (PPP, constant 2021 international $)
2. **Mean R&D expenditure** (% of GDP, 5-year average)
3. **Mean education spending** (% of GDP, 5-year average)
4. **Academic freedom index** (0-1 scale)
5. **Population**
6. **Post-Soviet country** (binary indicator for 15 former Soviet states)

---

## Key Findings

### Model Performance
- **R² = 0.8959** (explains 89.6% of variation)
- **Adjusted R² = 0.8938**
- **F-statistic = 421.92, p < 0.001** (highly significant)
- **Test R² = 0.8760** (excellent out-of-sample performance)
- **Test MAE = 576.86** researchers/million

### Significant Predictors (p < 0.01)

| Variable | Coefficient | Std. Error | t-statistic | p-value | 95% CI |
|----------|-------------|------------|-------------|---------|--------|
| **Mean R&D Expenditure (% GDP)** | 1,828.84 | 60.90 | 30.03 | < 0.001 *** | [1,709.0, 1,948.7] |
| **GDP per capita** | 0.0275 | 0.0022 | 12.47 | < 0.001 *** | [0.0231, 0.0318] |
| **Population** | -0.00003 | 0.000005 | -6.50 | < 0.001 *** | [-0.00004, -0.00002] |
| **Is Post-Soviet** | 526.06 | 128.91 | 4.08 | < 0.001 *** | [272.3, 779.8] |

### Non-Significant Predictors

| Variable | Coefficient | t-statistic | p-value |
|----------|-------------|-------------|---------|
| Academic Freedom Index | 250.57 | 1.42 | 0.156 |
| Mean Education Spending (% GDP) | -11.72 | -0.30 | 0.768 |

### Interpretation

1. **R&D Investment is Dominant**: Every 1% increase in R&D expenditure → +1,829 researchers/million (strongest effect)

2. **Economic Development Matters**: Every $1,000 increase in GDP per capita → +27.5 researchers/million

3. **Population Effect**: Larger countries have proportionally fewer researchers per capita (coordination/scaling challenges)

4. **Post-Soviet Legacy**: Former Soviet countries maintain +526 researchers/million advantage (persistent scientific culture)

5. **Education Spending Ineffective**: General education spending shows no significant relationship (targeted R&D investment matters more)

6. **Academic Freedom**: Not statistically significant (may work through indirect channels or be collinear with development)

---

## Project Structure

```
rnd-researchers/
├── requirements.txt         # Python dependencies
├── config.py               # Configuration and environment variables
├── data_loader.py          # Load datasets from Our World in Data
├── data_prep.py            # Data preprocessing and feature engineering
├── eda_augmented.py        # Exploratory data analysis
├── models.py               # Linear regression modeling
├── econometric_tests.py    # Econometric hypothesis testing (t-tests, F-test)
├── train_regression.py     # Main training pipeline
├── run_tests.py            # Quick script for hypothesis tests only
├── figures/                # Generated visualizations
└── README.md               # This file
```

---

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setup

```bash
# Clone the repository
git clone git@github.com:DanielyanEduard/regression-rnd-researchers.git
cd rnd-researchers

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Run Complete Analysis (with EDA and hypothesis tests)

```bash
python3 train_regression.py
```

This will:
1. Download 5 datasets from Our World in Data
2. Build augmented panel dataset (2019-2023)
3. Generate exploratory visualizations
4. Train linear regression model
5. Run econometric hypothesis tests (t-tests, F-test)
6. Evaluate on train/test split (80/20)

### Run Hypothesis Tests Only (faster)

```bash
python3 run_tests.py
```

Skips EDA and focuses on model training and econometric testing.

---

## Econometric Hypothesis Testing

The project implements two essential econometric tests:

### 1. t-tests for Individual Coefficients

**Hypotheses:**
- H₀: β_i = 0 (variable has no effect)
- H₁: β_i ≠ 0 (variable has significant effect)

**Results:**
- 4 out of 6 variables are highly significant (p < 0.01)
- R&D expenditure has the strongest effect (t = 30.03)

### 2. F-test for Overall Model

**Hypotheses:**
- H₀: All coefficients are zero (model has no explanatory power)
- H₁: At least one coefficient is non-zero

**Result:**
- F = 421.92, p < 0.001
- Model is highly significant

### Example Output

```
====================================================================================================
ECONOMETRIC HYPOTHESIS TESTING RESULTS
====================================================================================================

📊 MODEL SUMMARY
   Observations: 301
   Features: 6
   R²: 0.8959
   Adjusted R²: 0.8938

====================================================================================================
1. COEFFICIENT SIGNIFICANCE TESTS (t-tests)
====================================================================================================
   H₀: β_i = 0 (coefficient has no effect)
   H₁: β_i ≠ 0 (coefficient has significant effect)

Feature                                                 Coeff    Std.Err     t-stat    p-value Sig.      
----------------------------------------------------------------------------------------------------
GDP per capita, PPP (constant 2021 international $)     0.0275     0.0022    12.4674     0.0000 ***       
Mean R&D Expenditure (% GDP)                        1828.8427    60.8953    30.0326     0.0000 ***       
Mean Education Spending (% GDP)                      -11.7196    39.6592    -0.2955     0.7678           
Academic Freedom Index                               250.5698   176.1058     1.4228     0.1558           
Population                                            -0.0000     0.0000    -6.5028     0.0000 ***       
Is Post-Soviet                                       526.0573   128.9126     4.0807     0.0001 ***       

   Significance levels: *** p<0.01, ** p<0.05

====================================================================================================
2. OVERALL MODEL SIGNIFICANCE (F-test)
====================================================================================================
   H₀: All coefficients are zero (model has no explanatory power)
   H₁: At least one coefficient is non-zero

   F-statistic: 421.9196
   p-value: 0.000000
   df: (6, 294)
   ✅ RESULT: Reject H₀ - Model is statistically significant (p < 0.05)
```

---

## Data Sources

All datasets from [Our World in Data](https://ourworldindata.org/):

1. **Researchers in R&D** (per million people) with GDP per capita
2. **R&D Expenditure** (% of GDP)
3. **Public Education Spending** (% of GDP)
4. **Academic Freedom Index** (V-Dem Institute)
5. **Population** (with UN projections)

### Data Augmentation Approach

Instead of using a single year, the model leverages multiple years (2019-2023):
- Each country can contribute up to 5 observations (one per year)
- Features are calculated relative to each observation's year
- Increases effective sample size from ~65 to 301
- Improves model robustness and generalizability

---

## Technical Details

### Feature Engineering

For each country-year observation (year X):
- **GDP per capita**: Value in year X
- **R&D spending**: Mean over (X-4) to X (5-year average)
- **Education spending**: Mean over (X-4) to X (5-year average)
- **Academic freedom**: Value in year X
- **Population**: Value in year X
- **Post-Soviet flag**: Time-invariant (binary)

### Model Validation

- **Train/Test Split**: 80/20 (240 train, 61 test)
- **Metrics**: R², MAE, MSE

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
requests>=2.31.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
certifi>=2023.7.22
scipy>=1.9.0
statsmodels>=0.14.0
```

---

## Contributors

- [@albpiliposyan](https://github.com/albpiliposyan)
- [@ManeMkh](https://github.com/ManeMkh)


---

## License

This project is open source and available for educational purposes.

---


## Contact

For questions or collaboration: [@DanielyanEduard](https://github.com/DanielyanEduard)
