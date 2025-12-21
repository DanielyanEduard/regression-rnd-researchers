from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR


def _save_and_show(fig_path: Optional[Path] = None) -> None:
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_researcher_distributions(researcher_df: pd.DataFrame, save: bool = True) -> None:
    """Distribution plots for researcher dataset using histograms and KDE."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Year distribution
    axes[0, 0].hist(researcher_df["Year"], bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Year", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # Researchers distribution with KDE
    sns.histplot(
        data=researcher_df,
        x="Researchers in R&D (per million people)",
        kde=True,
        ax=axes[0, 1],
        bins=30,
        color="skyblue",
    )
    axes[0, 1].set_title(
        "Distribution of Researchers in R&D (per million people)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Researchers in R&D (per million people)")
    axes[0, 1].grid(True, alpha=0.3)

    # GDP per capita distribution with KDE
    sns.histplot(
        data=researcher_df,
        x="GDP per capita, PPP (constant 2021 international $)",
        kde=True,
        ax=axes[1, 0],
        bins=30,
        color="lightcoral",
    )
    axes[1, 0].set_title(
        "Distribution of GDP per capita", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("GDP per capita, PPP (constant 2021 international $)")
    axes[1, 0].grid(True, alpha=0.3)

    # Box plot comparison across time periods
    researcher_df_copy = researcher_df.copy()
    researcher_df_copy["Period"] = pd.cut(
        researcher_df_copy["Year"],
        bins=[1995, 2005, 2015, 2025],
        labels=["1996-2005", "2006-2015", "2016-2025"],
    )
    sns.boxplot(
        data=researcher_df_copy,
        x="Period",
        y="Researchers in R&D (per million people)",
        ax=axes[1, 1],
        hue="Period",
        palette="Set2",
        legend=False,
    )
    axes[1, 1].set_title(
        "Researchers in R&D by Time Period", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_ylabel("Researchers in R&D (per million people)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "researcher_distributions.png" if save else None
    _save_and_show(fig_path)


def plot_top_countries_researchers(
    researcher_df: pd.DataFrame, save: bool = True
) -> None:
    """
    Multi-panel plot showing top countries by researchers over time.
    Includes line plot and bar chart for recent year.
    """

    # Get 2022 data for top countries
    recent_data = researcher_df[researcher_df["Year"] == 2022].copy()
    top_countries = (
        recent_data.nlargest(10, "Researchers in R&D (per million people)")["Entity"]
        .unique()
        .tolist()
    )

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # Left: Line plot of top countries over time
    ax1 = fig.add_subplot(gs[0, 0])
    for country in top_countries:
        country_data = researcher_df[researcher_df["Entity"] == country]
        ax1.plot(
            country_data["Year"],
            country_data["Researchers in R&D (per million people)"],
            marker="o",
            label=country,
            linewidth=2,
            markersize=4,
        )
    ax1.set_title(
        "Top 10 Countries: Researchers in R&D Over Time",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Researchers in R&D (per million people)", fontsize=11)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Bar chart of top 15 in 2022
    ax2 = fig.add_subplot(gs[0, 1])
    top_15_2022 = recent_data.nlargest(15, "Researchers in R&D (per million people)")
    colors = plt.cm.viridis(np.linspace(0, 1, 15))
    ax2.barh(
        range(len(top_15_2022)),
        top_15_2022["Researchers in R&D (per million people)"],
        color=colors,
    )
    ax2.set_yticks(range(len(top_15_2022)))
    ax2.set_yticklabels(top_15_2022["Entity"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title(
        "Top 15 Countries in 2022", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Researchers in R&D (per million people)", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")

    fig_path = FIGURES_DIR / "top_countries_researchers.png" if save else None
    _save_and_show(fig_path)


def plot_spending_analysis(spending_df: pd.DataFrame, save: bool = True) -> None:
    """Enhanced R&D spending analysis with violin plot and trends."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Distribution with violin plot
    spending_df_copy = spending_df.copy()
    spending_df_copy["Period"] = pd.cut(
        spending_df_copy["Year"],
        bins=[1995, 2005, 2015, 2025],
        labels=["1996-2005", "2006-2015", "2016-2025"],
    )
    sns.violinplot(
        data=spending_df_copy,
        x="Period",
        y="Research and development expenditure (% of GDP)",
        ax=axes[0, 0],
        hue="Period",
        palette="muted",
        legend=False,
    )
    axes[0, 0].set_title(
        "R&D Expenditure Distribution by Period", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_ylabel("R&D expenditure (% of GDP)")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    # Add legend for violin plot
    from matplotlib.patches import Patch
    colors = sns.color_palette("muted", 3)
    legend_elements = [Patch(facecolor=colors[i], label=label) 
                      for i, label in enumerate(["1996-2005", "2006-2015", "2016-2025"])]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Histogram with KDE
    sns.histplot(
        data=spending_df,
        x="Research and development expenditure (% of GDP)",
        kde=True,
        ax=axes[0, 1],
        bins=40,
        color="teal",
        label="Distribution"
    )
    axes[0, 1].set_title(
        "Distribution of R&D Expenditure", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("R&D expenditure (% of GDP)")
    axes[0, 1].grid(True, alpha=0.3)
    # Add legend for histogram/KDE
    axes[0, 1].legend(loc='upper right', fontsize=9)

    # Top spenders in recent year
    recent_spending = spending_df[spending_df["Year"] == spending_df["Year"].max()]
    top_spenders = recent_spending.nlargest(
        15, "Research and development expenditure (% of GDP)"
    )
    axes[1, 0].barh(
        range(len(top_spenders)),
        top_spenders["Research and development expenditure (% of GDP)"],
        color=plt.cm.plasma(np.linspace(0.2, 0.8, 15)),
    )
    axes[1, 0].set_yticks(range(len(top_spenders)))
    axes[1, 0].set_yticklabels(top_spenders["Entity"], fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title(
        f"Top 15 R&D Spenders ({spending_df['Year'].max()})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("R&D expenditure (% of GDP)")
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    # Trend over time (global average)
    yearly_avg = (
        spending_df.groupby("Year")["Research and development expenditure (% of GDP)"]
        .mean()
        .reset_index()
    )
    axes[1, 1].plot(
        yearly_avg["Year"],
        yearly_avg["Research and development expenditure (% of GDP)"],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkblue",
        label="Average R&D Expenditure"
    )
    axes[1, 1].fill_between(
        yearly_avg["Year"],
        yearly_avg["Research and development expenditure (% of GDP)"],
        alpha=0.3,
        color="darkblue"
    )
    axes[1, 1].set_title(
        "Global Average R&D Expenditure Trend", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Average R&D expenditure (% of GDP)")
    axes[1, 1].grid(True, alpha=0.3)
    # Add legend for trend plot
    axes[1, 1].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "spending_analysis.png" if save else None
    _save_and_show(fig_path)


def plot_education_analysis(education_df: pd.DataFrame, save: bool = True) -> None:
    """Enhanced education spending analysis with multiple visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Distribution with violin plot
    education_df_copy = education_df.copy()
    education_df_copy["Period"] = pd.cut(
        education_df_copy["Year"],
        bins=[1995, 2005, 2015, 2025],
        labels=["1996-2005", "2006-2015", "2016-2025"],
    )
    sns.violinplot(
        data=education_df_copy,
        x="Period",
        y="Public spending on education as a share of GDP (historical and recent)",
        ax=axes[0, 0],
        hue="Period",
        palette="Set2",
        legend=False,
    )
    axes[0, 0].set_title(
        "Education Spending Distribution by Period", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_ylabel("Education spending (% of GDP)")
    axes[0, 0].set_xlabel("Time Period")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Histogram with KDE
    sns.histplot(
        data=education_df,
        x="Public spending on education as a share of GDP (historical and recent)",
        kde=True,
        ax=axes[0, 1],
        bins=40,
        color="orange",
    )
    axes[0, 1].set_title(
        "Distribution of Education Spending", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Education spending (% of GDP)")
    axes[0, 1].grid(True, alpha=0.3)

    # Top education spenders
    recent_education = education_df[
        education_df["Year"] == education_df["Year"].max()
    ]
    top_education = recent_education.nlargest(
        15, "Public spending on education as a share of GDP (historical and recent)"
    )
    axes[1, 0].barh(
        range(len(top_education)),
        top_education[
            "Public spending on education as a share of GDP (historical and recent)"
        ],
        color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, 15)),
    )
    axes[1, 0].set_yticks(range(len(top_education)))
    axes[1, 0].set_yticklabels(top_education["Entity"], fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title(
        f"Top 15 Education Spenders ({education_df['Year'].max()})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Education spending (% of GDP)")
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    # Global trend
    yearly_avg = (
        education_df.groupby("Year")[
            "Public spending on education as a share of GDP (historical and recent)"
        ]
        .mean()
        .reset_index()
    )
    axes[1, 1].plot(
        yearly_avg["Year"],
        yearly_avg[
            "Public spending on education as a share of GDP (historical and recent)"
        ],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkgreen",
    )
    axes[1, 1].fill_between(
        yearly_avg["Year"],
        yearly_avg[
            "Public spending on education as a share of GDP (historical and recent)"
        ],
        alpha=0.3,
    )
    axes[1, 1].set_title(
        "Global Average Education Spending Trend", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Average education spending (% of GDP)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "education_analysis.png" if save else None
    _save_and_show(fig_path)


def plot_correlation_heatmap(merged_df: pd.DataFrame, save: bool = True) -> None:
    """
    Correlation heatmap to visualize relationships between all numerical features.
    """

    # Select only numeric columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = merged_df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Correlation Heatmap of Features", fontsize=14, fontweight="bold", pad=20
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "correlation_heatmap.png" if save else None
    _save_and_show(fig_path)


def plot_regression_relationships(merged_df: pd.DataFrame, save: bool = True) -> None:
    """
    Scatter plots with regression lines to visualize relationships between 2022 
    researchers per million and key explanatory variables.
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: GDP per capita vs Researchers
    sns.regplot(
        data=merged_df,
        x="GDP per capita, PPP (constant 2021 international $)",
        y="Researchers in R&D (per million people) in 2022",
        ax=axes[0, 0],
        scatter_kws={"alpha": 0.6, "s": 60},
        line_kws={"color": "red", "linewidth": 2},
    )
    axes[0, 0].set_title(
        "Researchers in R&D (2022) vs. GDP per capita",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("GDP per capita (PPP, 2021 int'l $)")
    axes[0, 0].set_ylabel("Researchers in R&D (per million, 2022)")
    axes[0, 0].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    from scipy.stats import pearsonr

    # Get valid data (drop NaN)
    valid_data = merged_df[
        [
            "GDP per capita, PPP (constant 2021 international $)",
            "Researchers in R&D (per million people) in 2022",
        ]
    ].dropna()

    if len(valid_data) > 2:
        r_gdp, _ = pearsonr(
            valid_data["GDP per capita, PPP (constant 2021 international $)"],
            valid_data["Researchers in R&D (per million people) in 2022"],
        )
        axes[0, 0].text(
            0.05,
            0.95,
            f"R = {r_gdp:.3f}",
            transform=axes[0, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 2: Mean Researchers 2017-2021 vs Researchers 2022
    sns.regplot(
        data=merged_df,
        x="Mean Researchers R&D (2017-2021)",
        y="Researchers in R&D (per million people) in 2022",
        ax=axes[0, 1],
        scatter_kws={"alpha": 0.6, "s": 60, "color": "green"},
        line_kws={"color": "darkgreen", "linewidth": 2},
    )
    axes[0, 1].set_title(
        "Researchers in R&D (2022) vs. Mean Researchers (2017-2021)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Mean Researchers R&D (2017-2021)")
    axes[0, 1].set_ylabel("Researchers in R&D (per million, 2022)")
    axes[0, 1].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    valid_data_mean = merged_df[
        [
            "Mean Researchers R&D (2017-2021)",
            "Researchers in R&D (per million people) in 2022",
        ]
    ].dropna()

    if len(valid_data_mean) > 2:
        r_mean, _ = pearsonr(
            valid_data_mean["Mean Researchers R&D (2017-2021)"],
            valid_data_mean["Researchers in R&D (per million people) in 2022"],
        )
        axes[0, 1].text(
            0.05,
            0.95,
            f"R = {r_mean:.3f}",
            transform=axes[0, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 3: R&D Expenditure vs Researchers
    sns.regplot(
        data=merged_df,
        x="Mean Research and development expenditure (% of GDP) (2018-2022)",
        y="Researchers in R&D (per million people) in 2022",
        ax=axes[1, 0],
        scatter_kws={"alpha": 0.6, "s": 60, "color": "purple"},
        line_kws={"color": "darkviolet", "linewidth": 2},
    )
    axes[1, 0].set_title(
        "Researchers in R&D (2022) vs. R&D Expenditure (2018-2022)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Mean R&D expenditure (% of GDP, 2018-2022)")
    axes[1, 0].set_ylabel("Researchers in R&D (per million, 2022)")
    axes[1, 0].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    valid_data_rd = merged_df[
        [
            "Mean Research and development expenditure (% of GDP) (2018-2022)",
            "Researchers in R&D (per million people) in 2022",
        ]
    ].dropna()

    if len(valid_data_rd) > 2:
        r_rd, _ = pearsonr(
            valid_data_rd["Mean Research and development expenditure (% of GDP) (2018-2022)"],
            valid_data_rd["Researchers in R&D (per million people) in 2022"],
        )
        axes[1, 0].text(
            0.05,
            0.95,
            f"R = {r_rd:.3f}",
            transform=axes[1, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 4: Education Spending vs Researchers
    sns.regplot(
        data=merged_df,
        x="Mean public spending on education as a share of GDP (historical and recent) (2018-2022)",
        y="Researchers in R&D (per million people) in 2022",
        ax=axes[1, 1],
        scatter_kws={"alpha": 0.6, "s": 60, "color": "orange"},
        line_kws={"color": "darkorange", "linewidth": 2},
    )
    axes[1, 1].set_title(
        "Researchers in R&D (2022) vs. Education Spending (2018-2022)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Mean education spending (% of GDP, 2018-2022)")
    axes[1, 1].set_ylabel("Researchers in R&D (per million, 2022)")
    axes[1, 1].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    valid_data_edu = merged_df[
        [
            "Mean public spending on education as a share of GDP (historical and recent) (2018-2022)",
            "Researchers in R&D (per million people) in 2022",
        ]
    ].dropna()

    if len(valid_data_edu) > 2:
        r_edu, _ = pearsonr(
            valid_data_edu["Mean public spending on education as a share of GDP (historical and recent) (2018-2022)"],
            valid_data_edu["Researchers in R&D (per million people) in 2022"],
        )
        axes[1, 1].text(
            0.05,
            0.95,
            f"R = {r_edu:.3f}",
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    fig_path = FIGURES_DIR / "regression_relationships.png" if save else None
    _save_and_show(fig_path)


def run_basic_eda(
    researcher_df: pd.DataFrame,
    spending_df: pd.DataFrame,
    education_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Run the comprehensive EDA visualizations in sequence.
    """

    print("Generating distribution analysis for researchers...")
    plot_researcher_distributions(researcher_df, save=save)

    print("Generating top countries analysis...")
    plot_top_countries_researchers(researcher_df, save=save)

    print("Generating R&D spending analysis...")
    plot_spending_analysis(spending_df, save=save)

    print("Generating education spending analysis...")
    plot_education_analysis(education_df, save=save)

    print("Generating correlation heatmap...")
    plot_correlation_heatmap(merged_df, save=save)

    print("Generating regression relationship plots...")
    plot_regression_relationships(merged_df, save=save)

    print("\nEDA complete! All visualizations have been generated.")
