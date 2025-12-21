from functools import reduce
from typing import Tuple

import pandas as pd


def prepare_researcher_features(
    researcher_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the researcher feature engineering from the notebook:
      - keep Entity, Year, researchers per million, GDP per capita
      - drop missing rows
      - compute 2017–2021 mean researchers per million by Entity
      - extract 2022 researchers per million + GDP per capita by Entity
    """

    researcher_df = researcher_df[
        [
            "Entity",
            "Year",
            "Researchers in R&D (per million people)",
            "GDP per capita, PPP (constant 2021 international $)",
        ]
    ].dropna()

    previous_5years_researcher_df = researcher_df.loc[
        (researcher_df["Year"] >= 2017) & (researcher_df["Year"] <= 2021)
    ]
    mean_researchers_df = (
        previous_5years_researcher_df.groupby("Entity")[
            "Researchers in R&D (per million people)"
        ]
        .mean()
        .reset_index()
    )
    mean_researchers_df.rename(
        columns={
            "Researchers in R&D (per million people)": "Mean Researchers R&D (2017-2021)"
        },
        inplace=True,
    )

    last_year_researcher_df = researcher_df.loc[
        researcher_df["Year"] == 2022
    ][
        [
            "Entity",
            "Researchers in R&D (per million people)",
            "GDP per capita, PPP (constant 2021 international $)",
        ]
    ]
    last_year_researcher_df.rename(
        columns={
            "Researchers in R&D (per million people)": "Researchers in R&D (per million people) in 2022"
        },
        inplace=True,
    )

    return mean_researchers_df, last_year_researcher_df


def prepare_spending_features(spending_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce spending feature engineering:
      - filter 2018–2022
      - compute mean R&D expenditure (% of GDP) by Entity
    """

    spending_last_5years_df = spending_df[
        (spending_df["Year"] >= 2018) & (spending_df["Year"] <= 2022)
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
            "Research and development expenditure (% of GDP)": "Mean Research and development expenditure (% of GDP) (2018-2022)"
        },
        inplace=True,
    )
    return mean_spending_df


def prepare_education_features(education_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce education feature engineering:
      - filter 2018–2022
      - compute mean public spending on education as share of GDP by Entity
    """

    education_last_5years_df = education_df[
        (education_df["Year"] >= 2018) & (education_df["Year"] <= 2022)
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
            "Public spending on education as a share of GDP (historical and recent)": "Mean public spending on education as a share of GDP (historical and recent) (2018-2022)"
        },
        inplace=True,
    )
    return mean_education_df


def build_merged_dataset(
    researcher_df: pd.DataFrame, spending_df: pd.DataFrame, education_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the final modeling dataset, mirroring the notebook's merged_df.
    """

    mean_researchers_df, last_year_researcher_df = prepare_researcher_features(
        researcher_df
    )
    mean_spending_df = prepare_spending_features(spending_df)
    mean_education_df = prepare_education_features(education_df)

    dfs = [
        mean_researchers_df,
        last_year_researcher_df,
        mean_spending_df,
        mean_education_df,
    ]

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="Entity", how="inner"), dfs
    )
    return merged_df


