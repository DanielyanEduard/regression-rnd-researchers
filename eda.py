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
    """Top 15 countries by researchers and researchers count over years."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Top 15 countries by researchers in 2022
    recent_data = researcher_df[researcher_df["Year"] == 2022]
    top_15 = recent_data.nlargest(15, "Researchers in R&D (per million people)")
    colors = plt.cm.viridis(np.linspace(0, 1, 15))
    axes[0].barh(
        range(len(top_15)),
        top_15["Researchers in R&D (per million people)"],
        color=colors,
    )
    axes[0].set_yticks(range(len(top_15)))
    axes[0].set_yticklabels(top_15["Entity"], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title(
        "Top 15 Countries by Researchers in R&D (2022)", fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Researchers in R&D (per million people)")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Researchers count over years (global average)
    yearly_avg = (
        researcher_df.groupby("Year")["Researchers in R&D (per million people)"]
        .mean()
        .reset_index()
    )
    axes[1].plot(
        yearly_avg["Year"],
        yearly_avg["Researchers in R&D (per million people)"],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkblue",
        label="Average Researchers in R&D"
    )
    axes[1].fill_between(
        yearly_avg["Year"],
        yearly_avg["Researchers in R&D (per million people)"],
        alpha=0.3,
        color="darkblue"
    )
    axes[1].set_title(
        "Global Average Researchers in R&D Over Time", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Average Researchers in R&D (per million people)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=9)

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

    fig = plt.figure(figsize=(16, 8))
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
    ax1.legend(loc="upper left", fontsize=9)
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
    """R&D spending analysis showing top countries and global trend."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Top 15 R&D spenders
    recent_spending = spending_df[spending_df["Year"] == 2022]
    top_spenders = recent_spending.nlargest(
        15, "Research and development expenditure (% of GDP)"
    )
    axes[0].barh(
        range(len(top_spenders)),
        top_spenders["Research and development expenditure (% of GDP)"],
        color=plt.cm.plasma(np.linspace(0.2, 0.8, 15)),
    )
    axes[0].set_yticks(range(len(top_spenders)))
    axes[0].set_yticklabels(top_spenders["Entity"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(
        "Top 15 Countries by R&D Expenditure (2022)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("R&D expenditure (% of GDP)")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Trend over time (global average)
    yearly_avg = (
        spending_df.groupby("Year")["Research and development expenditure (% of GDP)"]
        .mean()
        .reset_index()
    )
    axes[1].plot(
        yearly_avg["Year"],
        yearly_avg["Research and development expenditure (% of GDP)"],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkblue",
        label="Average R&D Expenditure"
    )
    axes[1].fill_between(
        yearly_avg["Year"],
        yearly_avg["Research and development expenditure (% of GDP)"],
        alpha=0.3,
        color="darkblue"
    )
    axes[1].set_title(
        "Global Average R&D Expenditure Over Time", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Average R&D expenditure (% of GDP)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "spending_analysis.png" if save else None
    _save_and_show(fig_path)


def plot_education_analysis(education_df: pd.DataFrame, save: bool = True) -> None:
    """Education spending analysis showing top countries and global trend."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Top 15 education spenders
    recent_education = education_df[education_df["Year"] == 2022]
    top_education = recent_education.nlargest(
        15, "Public spending on education as a share of GDP (historical and recent)"
    )
    axes[0].barh(
        range(len(top_education)),
        top_education[
            "Public spending on education as a share of GDP (historical and recent)"
        ],
        color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, 15)),
    )
    axes[0].set_yticks(range(len(top_education)))
    axes[0].set_yticklabels(top_education["Entity"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(
        "Top 15 Countries by Education Spending (2022)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Education spending (% of GDP)")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Global trend
    yearly_avg = (
        education_df.groupby("Year")[
            "Public spending on education as a share of GDP (historical and recent)"
        ]
        .mean()
        .reset_index()
    )
    axes[1].plot(
        yearly_avg["Year"],
        yearly_avg[
            "Public spending on education as a share of GDP (historical and recent)"
        ],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkgreen",
        label="Average Education Spending"
    )
    axes[1].fill_between(
        yearly_avg["Year"],
        yearly_avg[
            "Public spending on education as a share of GDP (historical and recent)"
        ],
        alpha=0.3,
        color="darkgreen"
    )
    axes[1].set_title(
        "Global Average Education Spending Over Time", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Average education spending (% of GDP)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "education_analysis.png" if save else None
    _save_and_show(fig_path)


def plot_gdp_analysis(researcher_df: pd.DataFrame, save: bool = True) -> None:
    """Enhanced GDP per capita analysis with multiple visualizations."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Top GDP countries in recent year
    recent_gdp = researcher_df[researcher_df["Year"] == 2022]
    top_gdp = recent_gdp.nlargest(
        15, "GDP per capita, PPP (constant 2021 international $)"
    )
    axes[0].barh(
        range(len(top_gdp)),
        top_gdp["GDP per capita, PPP (constant 2021 international $)"],
        color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15)),
    )
    axes[0].set_yticks(range(len(top_gdp)))
    axes[0].set_yticklabels(top_gdp["Entity"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(
        f"Top 15 Countries by GDP per Capita (2022)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("GDP per capita (PPP, 2021 int'l $)")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Trend over time (global average)
    yearly_avg = (
        researcher_df.groupby("Year")["GDP per capita, PPP (constant 2021 international $)"]
        .mean()
        .reset_index()
    )
    axes[1].plot(
        yearly_avg["Year"],
        yearly_avg["GDP per capita, PPP (constant 2021 international $)"],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkred",
        label="Average GDP per Capita"
    )
    axes[1].fill_between(
        yearly_avg["Year"],
        yearly_avg["GDP per capita, PPP (constant 2021 international $)"],
        alpha=0.3,
        color="darkred"
    )
    axes[1].set_title(
        "Global Average GDP per Capita Trend", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Average GDP per capita (PPP, 2021 int'l $)")
    axes[1].grid(True, alpha=0.3)
    # Add legend for trend plot
    axes[1].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "gdp_analysis.png" if save else None
    _save_and_show(fig_path)


def plot_correlation_heatmap(merged_df: pd.DataFrame, target_year: int = 2022, save: bool = True) -> None:
    """
    Correlation heatmap to visualize relationships between all numerical features.
    """
    start_year = target_year - 4
    
    # Select only numeric columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Move target variable to the end
    target_col = f"Researchers in R&D (per million people) in {target_year}"
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
        numeric_cols.append(target_col)
    
    # Reorder the dataframe and compute correlation
    correlation_matrix = merged_df[numeric_cols].corr()
    
    # Create concise labels mapping
    label_mapping = {
        f"Researchers in R&D (per million people) in {target_year}": f"Researchers ({target_year})",
        "GDP per capita, PPP (constant 2021 international $)": "GDP per capita",
        f"Mean Research and development expenditure (% of GDP) ({start_year}-{target_year})": f"Mean R&D Spending ({start_year}-{target_year})",
        f"Mean public spending on education as a share of GDP (historical and recent) ({start_year}-{target_year})": f"Mean Edu Spending ({start_year}-{target_year})",
        f"Academic freedom index ({target_year})": f"Academic Freedom ({target_year})",
        f"Population ({target_year})": f"Population ({target_year})",
        "Is Post-Soviet": "Post-Soviet Flag",
    }
    
    # Rename correlation matrix columns and index
    correlation_matrix = correlation_matrix.rename(columns=label_mapping, index=label_mapping)

    plt.figure(figsize=(12, 10))
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
        f"Correlation Heatmap of Features ({target_year})", fontsize=14, fontweight="bold", pad=20
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"correlation_heatmap_{target_year}.png" if save else None
    _save_and_show(fig_path)


def plot_regression_relationships(merged_df: pd.DataFrame, target_year: int = 2022, save: bool = True) -> None:
    """
    Scatter plots with regression lines to visualize relationships between target year 
    researchers per million and key explanatory variables.
    """
    start_year = target_year - 4
    target_col = f"Researchers in R&D (per million people) in {target_year}"

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Plot 1: GDP per capita vs Researchers
    sns.regplot(
        data=merged_df,
        x="GDP per capita, PPP (constant 2021 international $)",
        y=target_col,
        ax=axes[0, 0],
        scatter_kws={"alpha": 0.6, "s": 60},
        line_kws={"color": "red", "linewidth": 2},
    )
    axes[0, 0].set_title(
        f"Researchers in R&D ({target_year}) vs. GDP per capita",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("GDP per capita (PPP, 2021 int'l $)")
    axes[0, 0].set_ylabel(f"Researchers in R&D (per million, {target_year})")
    axes[0, 0].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    from scipy.stats import pearsonr

    # Get valid data (drop NaN)
    valid_data = merged_df[
        [
            "GDP per capita, PPP (constant 2021 international $)",
            target_col,
        ]
    ].dropna()

    if len(valid_data) > 2:
        r_gdp, _ = pearsonr(
            valid_data["GDP per capita, PPP (constant 2021 international $)"],
            valid_data[target_col],
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

    # Plot 2: R&D Expenditure vs Researchers
    rd_col = f"Mean Research and development expenditure (% of GDP) ({start_year}-{target_year})"
    sns.regplot(
        data=merged_df,
        x=rd_col,
        y=target_col,
        ax=axes[0, 1],
        scatter_kws={"alpha": 0.6, "s": 60, "color": "purple"},
        line_kws={"color": "darkviolet", "linewidth": 2},
    )
    axes[0, 1].set_title(
        f"Researchers in R&D ({target_year}) vs. R&D Expenditure",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel(f"Mean R&D expenditure (% of GDP, {start_year}-{target_year})")
    axes[0, 1].set_ylabel(f"Researchers in R&D (per million, {target_year})")
    axes[0, 1].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    valid_data_rd = merged_df[[rd_col, target_col]].dropna()

    if len(valid_data_rd) > 2:
        r_rd, _ = pearsonr(
            valid_data_rd[rd_col],
            valid_data_rd[target_col],
        )
        axes[0, 1].text(
            0.05,
            0.95,
            f"R = {r_rd:.3f}",
            transform=axes[0, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 3: Education Spending vs Researchers
    edu_col = f"Mean public spending on education as a share of GDP (historical and recent) ({start_year}-{target_year})"
    sns.regplot(
        data=merged_df,
        x=edu_col,
        y=target_col,
        ax=axes[1, 0],
        scatter_kws={"alpha": 0.6, "s": 60, "color": "orange"},
        line_kws={"color": "darkorange", "linewidth": 2},
    )
    axes[1, 0].set_title(
        f"Researchers in R&D ({target_year}) vs. Education Spending",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel(f"Mean education spending (% of GDP, {start_year}-{target_year})")
    axes[1, 0].set_ylabel(f"Researchers in R&D (per million, {target_year})")
    axes[1, 0].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    valid_data_edu = merged_df[[edu_col, target_col]].dropna()

    if len(valid_data_edu) > 2:
        r_edu, _ = pearsonr(
            valid_data_edu[edu_col],
            valid_data_edu[target_col],
        )
        axes[1, 0].text(
            0.05,
            0.95,
            f"R = {r_edu:.3f}",
            transform=axes[1, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 4: Academic Freedom vs Researchers
    af_col = f"Academic freedom index ({target_year})"
    sns.regplot(
        data=merged_df,
        x=af_col,
        y=target_col,
        ax=axes[1, 1],
        scatter_kws={"alpha": 0.6, "s": 60, "color": "blue"},
        line_kws={"color": "darkblue", "linewidth": 2},
    )
    axes[1, 1].set_title(
        f"Researchers in R&D ({target_year}) vs. Academic Freedom",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel(f"Academic freedom index ({target_year})")
    axes[1, 1].set_ylabel(f"Researchers in R&D (per million, {target_year})")
    axes[1, 1].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    valid_data_af = merged_df[[af_col, target_col]].dropna()

    if len(valid_data_af) > 2:
        r_af, _ = pearsonr(
            valid_data_af[af_col],
            valid_data_af[target_col],
        )
        axes[1, 1].text(
            0.05,
            0.95,
            f"R = {r_af:.3f}",
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 5: Population vs Researchers (log scale for better visualization)
    pop_col = f"Population ({target_year})"
    # Create a copy with log-transformed population
    plot_data = merged_df[[pop_col, target_col]].copy()
    plot_data['log_pop'] = np.log10(plot_data[pop_col])
    
    sns.regplot(
        data=plot_data,
        x='log_pop',
        y=target_col,
        ax=axes[2, 0],
        scatter_kws={"alpha": 0.6, "s": 60, "color": "green"},
        line_kws={"color": "darkgreen", "linewidth": 2},
    )
    axes[2, 0].set_title(
        f"Researchers in R&D ({target_year}) vs. Population (log scale)",
        fontsize=12,
        fontweight="bold",
    )
    axes[2, 0].set_xlabel(f"Log10(Population) ({target_year})")
    axes[2, 0].set_ylabel(f"Researchers in R&D (per million, {target_year})")
    axes[2, 0].grid(True, alpha=0.3)

    # Calculate and display correlation coefficient
    valid_data_pop = plot_data[['log_pop', target_col]].dropna()

    if len(valid_data_pop) > 2:
        r_pop, _ = pearsonr(
            valid_data_pop['log_pop'],
            valid_data_pop[target_col],
        )
        axes[2, 0].text(
            0.05,
            0.95,
            f"R = {r_pop:.3f}",
            transform=axes[2, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 6: Post-Soviet Flag vs Researchers (box plot)
    post_soviet_data = merged_df[["Is Post-Soviet", target_col]].copy()
    post_soviet_data["Country Type"] = post_soviet_data["Is Post-Soviet"].map(
        {0: "Non Post-Soviet", 1: "Post-Soviet"}
    )
    
    sns.boxplot(
        data=post_soviet_data,
        x="Country Type",
        y=target_col,
        hue="Country Type",
        ax=axes[2, 1],
        palette=["lightblue", "salmon"],
        legend=False,
    )
    sns.stripplot(
        data=post_soviet_data,
        x="Country Type",
        y=target_col,
        ax=axes[2, 1],
        color="black",
        alpha=0.3,
        size=4,
    )
    axes[2, 1].set_title(
        f"Researchers in R&D ({target_year}) by Post-Soviet Status",
        fontsize=12,
        fontweight="bold",
    )
    axes[2, 1].set_xlabel("Country Type")
    axes[2, 1].set_ylabel(f"Researchers in R&D (per million, {target_year})")
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"regression_relationships_{target_year}.png" if save else None
    _save_and_show(fig_path)


def plot_academic_freedom_analysis(academic_freedom_df: pd.DataFrame, save: bool = True) -> None:
    """Academic freedom analysis showing top countries and global trend."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Identify the academic freedom column
    af_column = None
    for col in ["Academic freedom index (central estimate)", "Academic freedom index", "Academic Freedom Index"]:
        if col in academic_freedom_df.columns:
            af_column = col
            break
    
    if af_column is None:
        for col in academic_freedom_df.columns:
            if col not in ["Entity", "Code", "Year", "World region according to OWID"]:
                af_column = col
                break
    
    # Top 15 countries by academic freedom in 2022
    recent_af = academic_freedom_df[academic_freedom_df["Year"] == 2022]
    top_af = recent_af.nlargest(15, af_column)
    axes[0].barh(
        range(len(top_af)),
        top_af[af_column],
        color=plt.cm.YlGn(np.linspace(0.3, 0.9, 15)),
    )
    axes[0].set_yticks(range(len(top_af)))
    axes[0].set_yticklabels(top_af["Entity"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(
        "Top 15 Countries by Academic Freedom (2022)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Academic Freedom Index")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Trend over time (global average)
    yearly_avg = (
        academic_freedom_df.groupby("Year")[af_column]
        .mean()
        .reset_index()
    )
    axes[1].plot(
        yearly_avg["Year"],
        yearly_avg[af_column],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkblue",
        label="Average Academic Freedom"
    )
    axes[1].fill_between(
        yearly_avg["Year"],
        yearly_avg[af_column],
        alpha=0.3,
        color="darkblue"
    )
    axes[1].set_title(
        "Global Average Academic Freedom Over Time", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Average Academic Freedom Index")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "academic_freedom_analysis.png" if save else None
    _save_and_show(fig_path)


def plot_population_analysis(population_df: pd.DataFrame, save: bool = True) -> None:
    """Population analysis showing top countries and global trend."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Identify the population column
    pop_column = None
    for col in ["Population (historical)", "Population", "Population (historical estimates and projections)"]:
        if col in population_df.columns:
            pop_column = col
            break
    
    if pop_column is None:
        for col in population_df.columns:
            if col not in ["Entity", "Code", "Year", "World region according to OWID"]:
                pop_column = col
                break

    # Top 15 countries by population in 2022
    recent_pop = population_df[population_df["Year"] == 2022]
    top_pop = recent_pop.nlargest(15, pop_column)
    # Convert to millions for better readability
    pop_millions = top_pop[pop_column] / 1_000_000
    axes[0].barh(
        range(len(top_pop)),
        pop_millions,
        color=plt.cm.Blues(np.linspace(0.3, 0.9, 15)),
    )
    axes[0].set_yticks(range(len(top_pop)))
    axes[0].set_yticklabels(top_pop["Entity"], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(
        "Top 15 Countries by Population (2022)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Population (millions)")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Trend over time (global total)
    yearly_total = (
        population_df.groupby("Year")[pop_column]
        .sum()
        .reset_index()
    )
    # Convert to billions for better readability
    yearly_total["Population (billions)"] = yearly_total[pop_column] / 1_000_000_000
    
    axes[1].plot(
        yearly_total["Year"],
        yearly_total["Population (billions)"],
        marker="o",
        linewidth=2,
        markersize=5,
        color="darkgreen",
        label="Total World Population"
    )
    axes[1].fill_between(
        yearly_total["Year"],
        yearly_total["Population (billions)"],
        alpha=0.3,
        color="darkgreen"
    )
    axes[1].set_title(
        "World Population Over Time", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Total Population (billions)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "population_analysis.png" if save else None
    _save_and_show(fig_path)


def plot_post_soviet_comparison(merged_df: pd.DataFrame, target_year: int = 2022, save: bool = True) -> None:
    """Comparison of various metrics between Post-Soviet and non-Post-Soviet countries."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    start_year = target_year - 4
    
    # Prepare data with labels
    plot_df = merged_df.copy()
    plot_df["Country Type"] = plot_df["Is Post-Soviet"].map(
        {0: "Non Post-Soviet", 1: "Post-Soviet"}
    )
    
    # Plot 1: Researchers comparison
    target_col = f"Researchers in R&D (per million people) in {target_year}"
    sns.violinplot(
        data=plot_df,
        x="Country Type",
        y=target_col,
        hue="Country Type",
        ax=axes[0, 0],
        palette=["lightblue", "salmon"],
        legend=False,
    )
    axes[0, 0].set_title(
        f"Researchers Distribution ({target_year})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_ylabel(f"Researchers (per million)")
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: GDP per capita comparison
    sns.violinplot(
        data=plot_df,
        x="Country Type",
        y="GDP per capita, PPP (constant 2021 international $)",
        hue="Country Type",
        ax=axes[0, 1],
        palette=["lightblue", "salmon"],
        legend=False,
    )
    axes[0, 1].set_title(
        "GDP per Capita Distribution",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_ylabel("GDP per capita (PPP)")
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Academic Freedom comparison
    af_col = f"Academic freedom index ({target_year})"
    sns.violinplot(
        data=plot_df,
        x="Country Type",
        y=af_col,
        hue="Country Type",
        ax=axes[1, 0],
        palette=["lightblue", "salmon"],
        legend=False,
    )
    axes[1, 0].set_title(
        f"Academic Freedom Distribution ({target_year})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_ylabel("Academic Freedom Index")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: R&D Spending comparison
    rd_col = f"Mean Research and development expenditure (% of GDP) ({start_year}-{target_year})"
    sns.violinplot(
        data=plot_df,
        x="Country Type",
        y=rd_col,
        hue="Country Type",
        ax=axes[1, 1],
        palette=["lightblue", "salmon"],
        legend=False,
    )
    axes[1, 1].set_title(
        f"R&D Expenditure Distribution ({start_year}-{target_year})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_ylabel("R&D Expenditure (% of GDP)")
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"post_soviet_comparison_{target_year}.png" if save else None
    _save_and_show(fig_path)


def run_basic_eda(
    researcher_df: pd.DataFrame,
    spending_df: pd.DataFrame,
    education_df: pd.DataFrame,
    academic_freedom_df: pd.DataFrame,
    population_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    target_year: int = 2022,
    save: bool = True,
) -> None:
    """
    Run the comprehensive EDA visualizations in sequence.
    """

    print("Generating distribution analysis for researchers...")
    plot_researcher_distributions(researcher_df, save=save)

    print("Generating top countries analysis...")
    plot_top_countries_researchers(researcher_df, save=save)

    print("Generating GDP per capita analysis...")
    plot_gdp_analysis(researcher_df, save=save)

    print("Generating R&D spending analysis...")
    plot_spending_analysis(spending_df, save=save)

    print("Generating education spending analysis...")
    plot_education_analysis(education_df, save=save)

    print("Generating academic freedom analysis...")
    plot_academic_freedom_analysis(academic_freedom_df, save=save)

    print("Generating population analysis...")
    plot_population_analysis(population_df, save=save)

    print(f"Generating correlation heatmap for {target_year}...")
    plot_correlation_heatmap(merged_df, target_year=target_year, save=save)

    print(f"Generating regression relationship plots for {target_year}...")
    plot_regression_relationships(merged_df, target_year=target_year, save=save)

    print(f"Generating Post-Soviet comparison for {target_year}...")
    plot_post_soviet_comparison(merged_df, target_year=target_year, save=save)

    print("\nEDA complete! All visualizations have been generated.")
