# Feature Comparison: Before vs After

## BEFORE (4 features + target)

### Independent Variables:
1. Mean Researchers R&D (2017-2021)
2. GDP per capita, PPP (constant 2021 international $)
3. Mean Research and development expenditure (% of GDP) (2018-2022)
4. Mean public spending on education as a share of GDP (historical and recent) (2018-2022)

### Dependent Variable:
- Researchers in R&D (per million people) in 2022

---

## AFTER (6 features + target, dynamic years)

### Independent Variables:
1. ~~Mean Researchers R&D (2017-2021)~~ **[REMOVED]**
2. GDP per capita, PPP (constant 2021 international $)
3. Mean Research and development expenditure (% of GDP) (YEAR-4 to YEAR)
4. Mean public spending on education as a share of GDP (historical and recent) (YEAR-4 to YEAR)
5. **Academic freedom index (YEAR) [NEW]**
6. **Population (YEAR) [NEW]**
7. **Is Post-Soviet [NEW - Binary: 0 or 1]**

### Dependent Variable:
- Researchers in R&D (per million people) in YEAR
  - YEAR can be 2022, 2023, or 2024

---

## Key Changes Summary

### ❌ Removed (1 feature):
- **Mean Researchers R&D (2017-2021)**: Removed to avoid overfitting as it was highly correlated with target

### ✅ Added (3 features):
1. **Academic freedom index**: Measures the quality of academic environment
   - Source: Our World in Data
   - Range: 0-1 (higher = more freedom)
   
2. **Population**: Country size indicator
   - Source: Our World in Data
   - Used in log scale for better visualization
   
3. **Is Post-Soviet**: Binary indicator for post-Soviet countries
   - 15 countries: Russia, Ukraine, Belarus, Kazakhstan, Uzbekistan, Turkmenistan, 
     Kyrgyzstan, Tajikistan, Georgia, Armenia, Azerbaijan, Moldova, Lithuania, 
     Latvia, Estonia

### 🔄 Modified:
- Time windows now dynamic based on target year
- Support for multiple target years (2022, 2023, 2024)

---

## Model Improvements

### Net Change:
- **Before**: 4 features → **After**: 6 features
- **+50% more features** with better representation of research environment

### Expected Benefits:
1. **Academic Freedom**: Captures institutional quality and research environment
2. **Population**: Controls for country size effects
3. **Post-Soviet Flag**: Captures unique historical and institutional characteristics
4. **Removed Autocorrelation**: Eliminating mean researchers reduces overfitting

### Multi-Year Training:
- Can now compare model performance across years
- Automatically selects best year based on test R²
- Provides comprehensive comparison table

