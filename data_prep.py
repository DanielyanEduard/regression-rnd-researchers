from functools import reduce
from typing import Tuple

import pandas as pd


# List of post-Soviet countries
POST_SOVIET_COUNTRIES = [
    "Russia",
    "Ukraine",
    "Belarus",
    "Kazakhstan",
    "Uzbekistan",
    "Turkmenistan",
    "Kyrgyzstan",
    "Tajikistan",
    "Georgia",
    "Armenia",
    "Azerbaijan",
    "Moldova",
    "Lithuania",
    "Latvia",
    "Estonia",
]


def prepare_researcher_features(
    researcher_df: pd.DataFrame, target_year: int = 2022
) -> pd.DataFrame:
    """
    Prepare researcher features for a given target year.
    Only extract target year researchers per million + GDP per capita by Entity.
    """

    researcher_df = researcher_df[
        [
            "Entity",
            "Year",
            "Researchers in R&D (per million people)",
            "GDP per capita, PPP (constant 2021 international $)",
        ]
    ].dropna()

    last_year_researcher_df = researcher_df.loc[
        researcher_df["Year"] == target_year
    ][
        [
            "Entity",
            "Researchers in R&D (per million people)",
            "GDP per capita, PPP (constant 2021 international $)",
        ]
    ]
    last_year_researcher_df.rename(
        columns={
            "Researchers in R&D (per million people)": f"Researchers in R&D (per million people) in {target_year}"
        },
        inplace=True,
    )

    return last_year_researcher_df


def prepare_spending_features(spending_df: pd.DataFrame, target_year: int = 2022) -> pd.DataFrame:
    """
    Reproduce spending feature engineering:
      - filter (target_year-4) to target_year
      - compute mean R&D expenditure (% of GDP) by Entity
    """
    start_year = target_year - 4
    spending_last_5years_df = spending_df[
        (spending_df["Year"] >= start_year) & (spending_df["Year"] <= target_year)
    ]
    mean_spending_df = (
        spending_last_5years_df.groupby("Entity")[
            "Research and development expenditure (% of GDP)"
        ]
        .mean()
        .reset_index()
    )
    mean_spending_df.rename(
        columns={
            "Research and development expenditure (% of GDP)": f"Mean Research and development expenditure (% of GDP) ({start_year}-{target_year})"
        },
        inplace=True,
    )
    return mean_spending_df


def prepare_education_features(education_df: pd.DataFrame, target_year: int = 2022) -> pd.DataFrame:
    """
    Reproduce education feature engineering:
      - filter (target_year-4) to target_year
      - compute mean public spending on education as share of GDP by Entity
    """
    start_year = target_year - 4
    education_last_5years_df = education_df[
        (education_df["Year"] >= start_year) & (education_df["Year"] <= target_year)
    ]
    mean_education_df = (
        education_last_5years_df.groupby("Entity")[
            "Public spending on education as a share of GDP (historical and recent)"
        ]
        .mean()
        .reset_index()
    )
    mean_education_df.rename(
        columns={
            "Public spending on education as a share of GDP (historical and recent)": f"Mean public spending on education as a share of GDP (historical and recent) ({start_year}-{target_year})"
        },
        inplace=True,
    )
    return mean_education_df


def prepare_academic_freedom_features(academic_freedom_df: pd.DataFrame, target_year: int = 2022) -> pd.DataFrame:
    """
    Prepare academic freedom features:
      - Extract academic freedom index for target year by Entity
    """
    # Identify the academic freedom column (it may have different names)
    af_column = None
    possible_names = [
        "Academic freedom index (central estimate)",
        "Academic freedom index",
        "Academic Freedom Index",
    ]
    
    for col in possible_names:
        if col in academic_freedom_df.columns:
            af_column = col
            break
    
    if af_column is None:
        # If none of the expected names found, use the first numeric column after Entity, Code, Year
        for col in academic_freedom_df.columns:
            if col not in ["Entity", "Code", "Year", "World region according to OWID"]:
                af_column = col
                break
    
    if af_column is None:
        raise ValueError(f"Could not find academic freedom column. Available columns: {list(academic_freedom_df.columns)}")
    
    # Filter for the target year
    academic_freedom_year_df = academic_freedom_df[
        academic_freedom_df["Year"] == target_year
    ][["Entity", af_column]]
    
    academic_freedom_year_df.rename(
        columns={
            af_column: f"Academic freedom index ({target_year})"
        },
        inplace=True,
    )
    
    return academic_freedom_year_df


def prepare_population_features(population_df: pd.DataFrame, target_year: int = 2022) -> pd.DataFrame:
    """
    Prepare population features:
      - Extract population for target year by Entity
    """
    # Identify the population column (it may have different names)
    pop_column = None
    possible_names = [
        "Population (historical)",
        "Population",
        "Population (historical estimates and projections)",
    ]
    
    for col in possible_names:
        if col in population_df.columns:
            pop_column = col
            break
    
    if pop_column is None:
        # If none of the expected names found, use the first numeric column after Entity, Code, Year
        for col in population_df.columns:
            if col not in ["Entity", "Code", "Year", "World region according to OWID"]:
                pop_column = col
                break
    
    if pop_column is None:
        raise ValueError(f"Could not find population column. Available columns: {list(population_df.columns)}")
    
    # Filter for the target year
    population_year_df = population_df[
        population_df["Year"] == target_year
    ][["Entity", pop_column]]
    
    population_year_df.rename(
        columns={
            pop_column: f"Population ({target_year})"
        },
        inplace=True,
    )
    
    return population_year_df


def add_post_soviet_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary flag indicating whether a country is post-Soviet.
    """
    df["Is Post-Soviet"] = df["Entity"].isin(POST_SOVIET_COUNTRIES).astype(int)
    return df


def build_augmented_dataset(
    researcher_df: pd.DataFrame,
    spending_df: pd.DataFrame,
    education_df: pd.DataFrame,
    academic_freedom_df: pd.DataFrame,
    population_df: pd.DataFrame,
    min_year: int = 2019,
    max_year: int = 2023,
) -> pd.DataFrame:
    """
    Build an augmented dataset where each country contributes multiple rows
    across different years. This allows for data augmentation.
    
    For each country and year X where we have researcher data:
    - Target: Researchers in year X
    - Features computed relative to year X:
      - GDP per capita in year X
      - Mean R&D spending in (X-4) to X
      - Mean education spending in (X-4) to X
      - Academic freedom in year X
      - Population in year X
      - Is Post-Soviet flag
    
    Args:
        researcher_df: Researchers dataset
        spending_df: R&D spending dataset
        education_df: Education spending dataset
        academic_freedom_df: Academic freedom dataset
        population_df: Population dataset
        min_year: Minimum year to include (default: 2019)
        max_year: Maximum year to include (default: 2023)
    
    Returns:
        DataFrame with columns: Entity, Year, features, target
    """
    
    all_rows = []
    
    # Get years to process
    years_to_process = range(min_year, max_year + 1)
    
    for year in years_to_process:
        # Get researcher data for this year (target variable + GDP)
        researcher_year = researcher_df[researcher_df["Year"] == year][
            ["Entity", "Year", "Researchers in R&D (per million people)", 
             "GDP per capita, PPP (constant 2021 international $)"]
        ].dropna()
        
        if len(researcher_year) == 0:
            continue
        
        # Calculate spending features for (year-4) to year
        start_year = year - 4
        spending_period = spending_df[
            (spending_df["Year"] >= start_year) & (spending_df["Year"] <= year)
        ]
        mean_spending = (
            spending_period.groupby("Entity")["Research and development expenditure (% of GDP)"]
            .mean()
            .reset_index()
        )
        mean_spending.rename(
            columns={"Research and development expenditure (% of GDP)": "Mean R&D Expenditure (% GDP)"},
            inplace=True,
        )
        
        # Calculate education features for (year-4) to year
        education_period = education_df[
            (education_df["Year"] >= start_year) & (education_df["Year"] <= year)
        ]
        mean_education = (
            education_period.groupby("Entity")[
                "Public spending on education as a share of GDP (historical and recent)"
            ]
            .mean()
            .reset_index()
        )
        mean_education.rename(
            columns={
                "Public spending on education as a share of GDP (historical and recent)": 
                "Mean Education Spending (% GDP)"
            },
            inplace=True,
        )
        
        # Get academic freedom for this year
        academic_freedom_year = prepare_academic_freedom_features(academic_freedom_df, year)
        academic_freedom_year.rename(
            columns={f"Academic freedom index ({year})": "Academic Freedom Index"},
            inplace=True,
        )
        
        # Get population for this year
        population_year = prepare_population_features(population_df, year)
        population_year.rename(
            columns={f"Population ({year})": "Population"},
            inplace=True,
        )
        
        # Merge all features for this year
        year_data = researcher_year.copy()
        year_data = year_data.merge(mean_spending, on="Entity", how="left")
        year_data = year_data.merge(mean_education, on="Entity", how="left")
        year_data = year_data.merge(academic_freedom_year, on="Entity", how="left")
        year_data = year_data.merge(population_year, on="Entity", how="left")
        
        # Add post-Soviet flag
        year_data = add_post_soviet_flag(year_data)
        
        # Rename target variable to a consistent name
        year_data.rename(
            columns={"Researchers in R&D (per million people)": "Researchers per Million"},
            inplace=True,
        )
        
        all_rows.append(year_data)
    
    if not all_rows:
        return pd.DataFrame()
    
    # Combine all years
    augmented_df = pd.concat(all_rows, ignore_index=True)
    
    # Drop rows with any missing values
    rows_before = len(augmented_df)
    augmented_df = augmented_df.dropna()
    rows_after = len(augmented_df)
    
    if rows_before != rows_after:
        print(f"  Note: Dropped {rows_before - rows_after} rows with missing data")
        print(f"        ({rows_after} complete observations remaining)")
    
    return augmented_df


