# R&D Researchers

This project implements a structured Python pipeline for analyzing relationships between:
- **Researchers in R&D per million people**
- **GDP per capita (PPP)**
- **R&D expenditure as % of GDP**
- **Public spending on education as % of GDP**

using datasets from **[Our World in Data](https://ourworldindata.org/)** and a **linear regression model**.

> **📊 Real-World Data:** This project uses authentic, curated datasets from Our World in Data rather than synthetic or artificially generated data. This ensures the analysis reflects actual global trends and patterns in research, development, and education spending.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)

---

## Project Structure

```
rnd-researchers/
├── .env                 # Configuration with dataset URLs (included in repo)
├── .gitignore          # Git ignore rules
├── requirements.txt     # Python dependencies
├── config.py           # Configuration and environment variable loading
├── data_loader.py      # Functions to download/load datasets from OWID
├── data_prep.py        # Data filtering, aggregation, and merging
├── eda.py             # Exploratory data analysis and visualization
├── models.py          # Linear regression modeling and evaluation
├── train_regression.py # Main script - runs the complete pipeline
├── figures/           # Generated plots (created automatically)
└── README.md          # This file
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
git clone <your-repository-url>
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

The project includes a `.env` file with pre-configured dataset URLs from Our World in Data. No additional configuration is needed.

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
1. **Load datasets** from Our World in Data (using URLs from `.env`)
2. **Prepare and merge data** (filtering, aggregating by time periods)
3. **Generate EDA visualizations** (saved to `figures/` directory)
4. **Train a linear regression model** predicting researchers in R&D for 2022
5. **Perform K-fold cross-validation** on the training set
6. **Evaluate the model** on a held-out test set
7. **Print performance metrics** (R², MAE, MSE)

### Expected output

```
Loading datasets from OWID...
✓ Researchers dataset loaded
✓ Spending dataset loaded
✓ Education dataset loaded

Preparing data...
✓ Merged dataset created with 62 countries

Running EDA...
✓ Plots saved to figures/

Training model...
Full Model Performance:
  Intercept: 6.96
  Coefficients: [1.09, -0.002, 128.15, -4.84]
  R-squared: 0.9918

K-Fold Cross-Validation (5 folds):
  Average R-squared: 0.9897
  Average MAE: 171.49
  Average MSE: 65778.60

Test Set Performance:
  R-squared: 0.9859
  MAE: 205.21
  MSE: 84580.46
```

---

## Pipeline Overview

### 1. Data Loading (`data_loader.py`)
- Downloads three CSV files and their metadata from Our World in Data
- **Uses real-world, verified data** to ensure authenticity (no synthetic datasets)
- Uses the user-agent header specified in `.env`
- Datasets:
  - Researchers in R&D per million people vs GDP per capita
  - Research & Development expenditure (% of GDP)
  - Public spending on education as share of GDP

### 2. Data Preparation (`data_prep.py`)
- **Researcher data**: Calculates mean for 2017–2021 and extracts 2022 values
- **R&D spending**: Calculates mean for 2018–2022
- **Education spending**: Calculates mean for 2018–2022
- **Merges** all features into a single dataset by country/entity

### 3. Exploratory Data Analysis (`eda.py`)
- Generates boxplots for each dataset's key variables
- Creates comparison line plot (USA vs Russia researchers)
- Produces scatter plots showing relationships between features
- Saves all visualizations to `figures/` directory

### 4. Modeling (`models.py`)

**Target variable:**
- `Researchers in R&D (per million people) in 2022`

**Features:**
- Mean Researchers R&D (2017-2021)
- GDP per capita, PPP (constant 2021 international $)
- Mean R&D expenditure (% of GDP) (2018-2022)
- Mean public spending on education (% of GDP) (2018-2022)

**Model evaluation includes:**
- Full dataset regression with coefficients and R²
- 80/20 train-test split
- 5-fold cross-validation on training data
- Final evaluation on held-out test set
- Metrics: R², Mean Absolute Error (MAE), Mean Squared Error (MSE)

---

## Project Features

✅ **Real-world data** – Uses authentic datasets from Our World in Data (not synthetic)  
✅ Modular, maintainable code structure  
✅ Environment-based configuration  
✅ Automated data loading from remote sources  
✅ Comprehensive exploratory data analysis  
✅ Rigorous model validation (K-fold CV + test set)  
✅ Reproducible results with fixed random seeds  
✅ Production-ready Python project structure

---

## Contributors

- [@albpiliposyan](https://github.com/albpiliposyan)

---

## License

This project is open source and available for educational purposes.

---

## Contributing

Feel free to open issues or submit pull requests for improvements!


