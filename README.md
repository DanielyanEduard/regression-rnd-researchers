# R&D Researchers - Augmented Dataset Regression Analysis

This project implements a structured Python pipeline for analyzing relationships between:
- **Researchers in R&D per million people**
- **GDP per capita (PPP)**
- **R&D expenditure as % of GDP**
- **Public spending on education as % of GDP**
- **Academic freedom index** ⭐ NEW
- **Population** ⭐ NEW
- **Post-Soviet country status** ⭐ NEW

using datasets from **[Our World in Data](https://ourworldindata.org/)** and a **linear regression model** with an **augmented dataset approach (2019-2023)**.

> **📊 Real-World Data:** This project uses authentic, curated datasets from Our World in Data rather than synthetic or artificially generated data. This ensures the analysis reflects actual global trends and patterns in research, development, and education spending.

> **🎯 Data Augmentation:** The model uses an augmented dataset approach where each country contributes multiple observations across years (2019-2023), with features calculated relative to each year. This increases the effective dataset size and improves model robustness.

---

## Table of Contents
- [What's New](#whats-new)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Model Features](#model-features)

---

## What's New

### Recent Enhancements (v3.0 - Augmented Dataset)

1. **✅ Augmented Dataset Approach**: Each country contributes multiple observations (2019-2023), increasing dataset size from ~65 to 301 observations
2. **✅ Dynamic Feature Calculation**: Features computed relative to each observation's year X (e.g., R&D spending in (X-4) to X)
3. **✅ Improved Model Performance**: Test R² = 0.8760, MAE = 576.86 researchers/million
4. **✅ Global Visualizations**: All visualizations show overall patterns across years, not year-specific analysis
5. **✅ Added Academic Freedom Index**: Captures institutional quality and research environment
6. **✅ Added Population Feature**: Controls for country size effects  
7. **✅ Added Post-Soviet Flag**: Binary indicator for 15 post-Soviet countries

**Dataset**: 301 observations from 87 countries across 2019-2023 with data augmentation.

See [AUGMENTED_APPROACH.md](AUGMENTED_APPROACH.md) for details on the augmented dataset approach and [GLOBAL_VISUALIZATIONS.md](GLOBAL_VISUALIZATIONS.md) for visualization details.

---

## Project Structure

```
rnd-researchers/
├── .env                     # Configuration with dataset URLs (optional)
├── .gitignore              # Git ignore rules
├── requirements.txt         # Python dependencies
├── config.py               # Configuration and environment variable loading
├── data_loader.py          # Functions to download/load 5 datasets from OWID
├── data_prep.py            # Data filtering, aggregation, merging + post-Soviet flag
├── eda.py                  # EDA and visualization (10+ plots)
├── models.py               # Linear regression modeling and evaluation
├── train_regression.py     # Main script - runs multi-year training pipeline
├── figures/                # Generated plots (15+ visualizations)
├── CHANGES_SUMMARY.md      # Detailed documentation of all changes
├── FEATURE_COMPARISON.md   # Before/after feature comparison
├── VISUALIZATIONS.md       # Complete visualization guide
└── README.md               # This file
```

---

## Prerequisites

- **Python 3.7+**
- **pip** (Python package installer)
- **Git** (to clone the repository)

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:DanielyanEduard/regression-rnd-researchers.git
cd rnd-researchers
```

### 2. Create and activate a virtual environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

**After activating the virtual environment:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for macOS users:** If `pip` doesn't work, use `pip3` instead:
> ```bash
> pip3 install --upgrade pip
> pip3 install -r requirements.txt
> ```

The project includes default dataset URLs in `config.py`. You can optionally create a `.env` file to override these URLs.

**Optional .env configuration:**
```bash
RESEARCHERS_CSV_URL=<custom-url>
SPENDING_CSV_URL=<custom-url>
EDUCATION_CSV_URL=<custom-url>
# Academic Freedom and Population URLs have defaults
```

---

## Usage

### Run the complete pipeline

With the virtual environment activated and dependencies installed:

```bash
python train_regression.py
```

> **Note for macOS users:** If `python` doesn't work, use `python3`:
> ```bash
> python3 train_regression.py
> ```

This script will:
1. **Load 5 datasets** from Our World in Data:
   - Researchers in R&D per million people
   - R&D expenditure (% of GDP)
   - Public spending on education (% of GDP)
   - Academic freedom index ⭐ NEW
   - Population ⭐ NEW
2. **Build augmented dataset** for years 2019-2023:
   - For each country-year combination with researcher data
   - Calculate features relative to that year X:
     - GDP per capita in year X
     - Mean R&D spending in (X-4) to X
     - Mean education spending in (X-4) to X
     - Academic freedom in year X
     - Population in year X
     - Post-Soviet flag (constant)
3. **Generate global EDA visualizations** (4 comprehensive plots saved to `figures/`)
   - Overall distributions across all years
   - Global feature relationships
   - Correlation analysis
   - Top countries ranking
4. **Train single linear regression model** on augmented dataset (301 observations)
5. **Perform K-fold cross-validation** on training set
6. **Evaluate on held-out test set** (80/20 split)
7. **Compare performance across years** and identify best model
8. **Print detailed performance metrics** (R², MAE, MSE) for each year

### Expected output

```
Loading datasets...
✓ Datasets loaded successfully!

================================================================================
TRAINING MODEL FOR YEAR 2022
================================================================================
Merged dataset shape: (XX, 7)

Running EDA for 2022 and saving figures to the figures/ directory...
Generating distribution analysis for researchers...
Generating top countries analysis...
[... EDA output ...]

Full-data model (in-sample) for 2022:
  Intercept: XXX.XX
  Coefficients:
    GDP per capita, PPP (constant 2021 international $): X.XXXX
    Mean Research and development expenditure (% of GDP) (2018-2022): X.XXXX
    Mean public spending on education (% of GDP) (2018-2022): X.XXXX
    Academic freedom index (2022): X.XXXX
    Population (2022): X.XXXX
    Is Post-Soviet: X.XXXX
  R-squared: 0.XXXX
  MAE: XXX.XX
  MSE: XXXXX.XX

[... Similar output for 2023 and 2024 ...]

================================================================================
SUMMARY COMPARISON ACROSS YEARS
================================================================================
Year     Size     Full R²      CV R²        Test R²      Test MAE    
--------------------------------------------------------------------------------
2022     XX       0.XXXX       0.XXXX       0.XXXX       XXX.XX
2023     XX       0.XXXX       0.XXXX       0.XXXX       XXX.XX
2024     XX       0.XXXX       0.XXXX       0.XXXX       XXX.XX

================================================================================
BEST MODEL: Year XXXX with Test R² = 0.XXXX
================================================================================
```

---

## Pipeline Overview

### 1. Data Loading (`data_loader.py`)
- Downloads **5 CSV files** and their metadata from Our World in Data
- **Uses real-world, verified data** to ensure authenticity (no synthetic datasets)
- Uses the user-agent header specified in config
- Datasets:
  - Researchers in R&D per million people vs GDP per capita
  - Research & Development expenditure (% of GDP)
  - Public spending on education as share of GDP
  - **Academic freedom index** ⭐ NEW
  - **Population (with UN projections)** ⭐ NEW

### 2. Data Preparation (`data_prep.py`)
- **Researcher data**: Extracts values for target year (2022/2023/2024) with GDP per capita
- **R&D spending**: Calculates mean for (target_year - 4) to target_year
- **Education spending**: Calculates mean for (target_year - 4) to target_year
- **Academic freedom**: Extracts values for target year ⭐ NEW
- **Population**: Extracts values for target year ⭐ NEW
- **Post-Soviet flag**: Binary indicator for 15 post-Soviet countries ⭐ NEW
- **Merges** all features into a single dataset by country/entity

### 3. Exploratory Data Analysis (`eda.py`)
Generates **15+ visualizations** including:
- Researcher distributions and top countries analysis
- GDP, R&D spending, and education spending analysis
- **Academic freedom analysis** ⭐ NEW
- **Population analysis** ⭐ NEW
- Correlation heatmaps (per year)
- Regression relationships (6 plots including new features)
- **Post-Soviet country comparisons** ⭐ NEW

See [VISUALIZATIONS.md](VISUALIZATIONS.md) for complete guide.

### 4. Modeling (`models.py`)

**Target variable:**
- `Researchers in R&D (per million people) in [YEAR]`
  - YEAR can be 2022, 2023, or 2024

**Features (6 total):**
1. GDP per capita, PPP (constant 2021 international $)
2. Mean R&D expenditure (% of GDP) (YEAR-4 to YEAR)
3. Mean public spending on education (% of GDP) (YEAR-4 to YEAR)
4. **Academic freedom index (YEAR)** ⭐ NEW
5. **Population (YEAR)** ⭐ NEW
6. **Is Post-Soviet** ⭐ NEW

**Model evaluation includes:**
- Full dataset regression with coefficients and R² for each year
- 80/20 train-test split
- 5-fold cross-validation on training data
- Final evaluation on held-out test set
- **Comparison across years to select best model** ⭐ NEW
- Metrics: R², Mean Absolute Error (MAE), Mean Squared Error (MSE)

---

## Model Features

### Current Features (6 features)

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| GDP per capita | Continuous | OWID | Economic indicator (PPP, 2021 int'l $) |
| R&D Expenditure | Continuous | OWID | Mean R&D spending (% of GDP, 5-year avg) |
| Education Spending | Continuous | OWID | Mean education spending (% of GDP, 5-year avg) |
| Academic Freedom ⭐ | Continuous | OWID | Academic freedom index (0-1 scale) |
| Population ⭐ | Continuous | OWID | Country population |
| Is Post-Soviet ⭐ | Binary | Custom | Post-Soviet country flag (0/1) |

### Post-Soviet Countries (15 total)
Russia, Ukraine, Belarus, Kazakhstan, Uzbekistan, Turkmenistan, Kyrgyzstan, Tajikistan, Georgia, Armenia, Azerbaijan, Moldova, Lithuania, Latvia, Estonia

### Model Improvements Over Previous Version
- **+50% more features** (4 → 6 features)
- **Removed autocorrelation** by eliminating mean researchers feature
- **Better institutional representation** via academic freedom index
- **Country size control** via population feature
- **Historical context** via post-Soviet flag
- **Multi-year comparison** to select optimal target year

---

## Project Features

✅ **Real-world data** – Uses authentic datasets from Our World in Data (not synthetic)  
✅ **Multi-year support** – Train and compare models for 2022, 2023, 2024  
✅ **Enhanced features** – 6 features including academic freedom, population, post-Soviet status  
✅ Modular, maintainable code structure  
✅ Environment-based configuration  
✅ Automated data loading from remote sources  
✅ **Comprehensive EDA** – 15+ visualizations  
✅ Rigorous model validation (K-fold CV + test set)  
✅ Reproducible results with fixed random seeds  
✅ Production-ready Python project structure  
✅ **Detailed documentation** – Multiple MD files with complete guides

---

## Contributors

- [@DanielyanEduard](https://github.com/DanielyanEduard)

---

## License

This project is open source and available for educational purposes.

---

## Contributing

Feel free to open issues or submit pull requests for improvements!


