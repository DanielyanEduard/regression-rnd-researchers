# Visualization Outputs

## Overview
The updated EDA system generates comprehensive visualizations for all features including the new ones.

## Generated Files (in `figures/` directory)

### 1. Researcher Analysis
**Files**: 
- `researcher_distributions.png`
- `top_countries_researchers.png`

**Content**:
- Top 15 countries by researchers in R&D (2022)
- Global average researchers over time
- Top 10 countries trend lines over time
- Bar chart of top 15 countries in 2022

### 2. GDP Analysis
**File**: `gdp_analysis.png`

**Content**:
- Top 15 countries by GDP per capita (2022)
- Global average GDP per capita trend over time

### 3. R&D Spending Analysis
**File**: `spending_analysis.png`

**Content**:
- Top 15 countries by R&D expenditure (2022)
- Global average R&D expenditure over time

### 4. Education Spending Analysis
**File**: `education_analysis.png`

**Content**:
- Top 15 countries by education spending (2022)
- Global average education spending over time

### 5. Academic Freedom Analysis [NEW]
**File**: `academic_freedom_analysis.png`

**Content**:
- Top 15 countries by academic freedom index (2022)
- Global average academic freedom trend over time
- Shows evolution of academic freedom worldwide

### 6. Population Analysis [NEW]
**File**: `population_analysis.png`

**Content**:
- Top 15 countries by population (2022)
- World population growth over time
- Population displayed in millions/billions for readability

### 7. Correlation Heatmap
**Files**: 
- `correlation_heatmap_2022.png`
- `correlation_heatmap_2023.png` (if training for 2023)
- `correlation_heatmap_2024.png` (if training for 2024)

**Content**:
- Correlation matrix of all features including:
  - Researchers (target)
  - GDP per capita
  - R&D Spending
  - Education Spending
  - Academic Freedom [NEW]
  - Population [NEW]
  - Post-Soviet Flag [NEW]
- Shows Pearson correlation coefficients
- Color-coded (red = positive, blue = negative)

### 8. Regression Relationships
**Files**: 
- `regression_relationships_2022.png`
- `regression_relationships_2023.png` (if training for 2023)
- `regression_relationships_2024.png` (if training for 2024)

**Content**: 6 scatter plots with regression lines
1. **GDP per capita vs Researchers**
   - Shows economic impact on research capacity
   
2. **R&D Expenditure vs Researchers**
   - Shows investment effect on researcher numbers
   
3. **Education Spending vs Researchers**
   - Shows education investment impact
   
4. **Academic Freedom vs Researchers** [NEW]
   - Shows relationship between academic environment and researchers
   
5. **Population vs Researchers (log scale)** [NEW]
   - Shows country size effect
   - Uses log scale for better visualization
   
6. **Post-Soviet Status vs Researchers** [NEW]
   - Box plot comparing distributions
   - Shows effect of post-Soviet status

Each plot includes:
- Scatter points (semi-transparent for overlap visibility)
- Regression line
- Pearson correlation coefficient (R value)
- Grid for easy reading

### 9. Post-Soviet Comparison [NEW]
**Files**: 
- `post_soviet_comparison_2022.png`
- `post_soviet_comparison_2023.png` (if training for 2023)
- `post_soviet_comparison_2024.png` (if training for 2024)

**Content**: 4 violin plots comparing distributions
1. **Researchers Distribution**
   - Post-Soviet vs Non Post-Soviet countries
   
2. **GDP per Capita Distribution**
   - Economic comparison between groups
   
3. **Academic Freedom Distribution**
   - Academic environment comparison
   
4. **R&D Expenditure Distribution**
   - Investment comparison

Violin plots show:
- Full distribution shape
- Median and quartiles
- Comparison between two groups (color-coded)

## Visualization Summary

### Total Files Generated:
- **Minimum**: 11 files (if only training for 2022)
- **Maximum**: 15 files (if training for 2022, 2023, and 2024)

### New Visualizations Added:
1. Academic Freedom Analysis (1 file)
2. Population Analysis (1 file)
3. Enhanced Regression Relationships (+2 plots per year)
4. Post-Soviet Comparison (1 file per year)
5. Updated Correlation Heatmap (includes 3 new features)

### Color Schemes:
- **Researchers**: Blues/Viridis
- **GDP**: Red-Yellow-Green
- **R&D Spending**: Plasma/Purple
- **Education**: Yellow-Orange-Red
- **Academic Freedom**: Yellow-Green
- **Population**: Blues
- **Post-Soviet**: Light Blue (Non) vs Salmon (Post-Soviet)

### Key Features:
- High resolution (300 DPI)
- Professional styling
- Clear labels and titles
- Grid lines for readability
- Correlation coefficients where applicable
- Consistent color schemes across related plots

## Usage

All visualizations are automatically generated when running:
```bash
python3 train_regression.py
```

To generate only EDA without training:
```python
from data_loader import load_all_datasets
from data_prep import build_merged_dataset
from eda import run_basic_eda

# Load data
(researcher_df, spending_df, education_df, 
 academic_freedom_df, population_df) = load_all_datasets()

# Build merged dataset
merged_df = build_merged_dataset(
    researcher_df, spending_df, education_df, 
    academic_freedom_df, population_df, 
    target_year=2022
)

# Generate visualizations
run_basic_eda(
    researcher_df, spending_df, education_df,
    academic_freedom_df, population_df, merged_df,
    target_year=2022, save=True
)
```

