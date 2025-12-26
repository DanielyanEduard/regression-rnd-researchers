"""
EDA functions for augmented dataset - Global Analysis.
Focuses on overall patterns across years, not year-specific analysis.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from config import FIGURES_DIR


def _save_and_show(fig_path: Optional[Path] = None) -> None:
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_overall_distributions(merged_df: pd.DataFrame, save: bool = True) -> None:
    """Show overall distributions and relationships (not year-specific)."""
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    # Plot 1: Overall distribution of researchers (2019-2023 data)
    axes[0, 0].hist(merged_df['Researchers per Million'], bins=30, 
                     color='steelblue', alpha=0.7, edgecolor='black')
    res_mean_2019_2023 = merged_df['Researchers per Million'].mean()
    res_median_2019_2023 = merged_df['Researchers per Million'].median()
    axes[0, 0].axvline(res_mean_2019_2023, 
                        color='red', linestyle='--', linewidth=2, 
                        label=f"Mean (2019-2023): {res_mean_2019_2023:.0f}")
    axes[0, 0].axvline(res_median_2019_2023, 
                        color='green', linestyle='--', linewidth=2, 
                        label=f"Median (2019-2023): {res_median_2019_2023:.0f}")
    axes[0, 0].set_title('Overall Distribution of Researchers (2019-2023)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Researchers per Million')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Education Spending distribution (averaged over past 5 years for each obs)
    # Note: Each observation already has mean of (year-4 to year), this shows distribution of those means
    axes[0, 1].hist(merged_df['Mean Education Spending (% GDP)'], 
                     bins=30, color='orange', alpha=0.7, edgecolor='black')
    edu_mean = merged_df['Mean Education Spending (% GDP)'].mean()  # Mean of means (2019-2023)
    edu_median = merged_df['Mean Education Spending (% GDP)'].median()  # Median of means (2019-2023)
    axes[0, 1].axvline(edu_mean, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean (2019-2023): {edu_mean:.2f}%')
    axes[0, 1].axvline(edu_median, color='green', linestyle='--', linewidth=2, 
                        label=f'Median (2019-2023): {edu_median:.2f}%')
    axes[0, 1].set_title('Education Spending Distribution (2019-2023)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Mean Education Spending (past 5 years per obs, % GDP)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: R&D Expenditure distribution (averaged over past 5 years for each obs)
    # Note: Each observation already has mean of (year-4 to year), this shows distribution of those means
    axes[0, 2].hist(merged_df['Mean R&D Expenditure (% GDP)'], 
                     bins=30, color='purple', alpha=0.7, edgecolor='black')
    rd_mean = merged_df['Mean R&D Expenditure (% GDP)'].mean()  # Mean of means (2019-2023)
    rd_median = merged_df['Mean R&D Expenditure (% GDP)'].median()  # Median of means (2019-2023)
    axes[0, 2].axvline(rd_mean, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean (2019-2023): {rd_mean:.2f}%')
    axes[0, 2].axvline(rd_median, color='green', linestyle='--', linewidth=2, 
                        label=f'Median (2019-2023): {rd_median:.2f}%')
    axes[0, 2].set_title('R&D Expenditure Distribution (2019-2023)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Mean R&D Expenditure (past 5 years per obs, % GDP)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Academic Freedom distribution (2019-2023 data)
    axes[1, 0].hist(merged_df['Academic Freedom Index'], 
                     bins=30, color='orange', alpha=0.7, edgecolor='black')
    af_mean = merged_df['Academic Freedom Index'].mean()  # Mean across 2019-2023
    af_median = merged_df['Academic Freedom Index'].median()  # Median across 2019-2023
    axes[1, 0].axvline(af_mean, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean (2019-2023): {af_mean:.3f}')
    axes[1, 0].axvline(af_median, color='green', linestyle='--', linewidth=2, 
                        label=f'Median (2019-2023): {af_median:.3f}')
    axes[1, 0].set_title('Academic Freedom Distribution (2019-2023)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Academic Freedom Index (current year)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Post-Soviet comparison (boxplot)
    post_soviet_data = merged_df.copy()
    post_soviet_data['Country Type'] = post_soviet_data['Is Post-Soviet'].map(
        {0: 'Non Post-Soviet', 1: 'Post-Soviet'}
    )
    
    sns.boxplot(
        data=post_soviet_data,
        x='Country Type',
        y='Researchers per Million',
        hue='Country Type',
        ax=axes[1, 1],
        palette=['lightblue', 'salmon'],
        legend=False
    )
    
    ps_mean = merged_df[merged_df['Is Post-Soviet'] == 1]['Researchers per Million'].mean()
    non_ps_mean = merged_df[merged_df['Is Post-Soviet'] == 0]['Researchers per Million'].mean()
    diff = ps_mean - non_ps_mean
    
    axes[1, 1].set_title('Post-Soviet vs Non Post-Soviet (2019-2023)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Researchers per Million')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    # axes[1, 1].text(
    #     0.5, 0.95,
    #     f'Post-Soviet avg: +{diff:.0f} researchers/million',
    #     transform=axes[1, 1].transAxes,
    #     ha='center',
    #     va='top',
    #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    #     fontsize=10,
    #     fontweight='bold'
    # )
    
    # Plot 6: GDP per capita distribution (2019-2023 data)
    axes[1, 2].hist(merged_df['GDP per capita, PPP (constant 2021 international $)'], 
                     bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('GDP per Capita Distribution (2019-2023)', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('GDP per capita (PPP, constant 2021 $, current year)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add mean and median lines
    gdp_mean = merged_df['GDP per capita, PPP (constant 2021 international $)'].mean()  # Mean across 2019-2023
    gdp_median = merged_df['GDP per capita, PPP (constant 2021 international $)'].median()  # Median across 2019-2023
    axes[1, 2].axvline(gdp_mean, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean (2019-2023): ${gdp_mean:,.0f}')
    axes[1, 2].axvline(gdp_median, color='blue', linestyle='--', linewidth=2, 
                        label=f'Median (2019-2023): ${gdp_median:,.0f}')
    axes[1, 2].legend(fontsize=9)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "global_distributions.png" if save else None
    _save_and_show(fig_path)


def plot_feature_relationships(merged_df: pd.DataFrame, save: bool = True) -> None:
    """Scatter plots showing global relationships between features and target."""
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    # Common styling
    scatter_kws = {'alpha': 0.5, 's': 40, 'edgecolors': 'black', 'linewidths': 0.5}
    
    # Plot 1: GDP vs Researchers
    sns.regplot(
        data=merged_df,
        x='GDP per capita, PPP (constant 2021 international $)',
        y='Researchers per Million',
        ax=axes[0, 0],
        scatter_kws=scatter_kws,
        line_kws={'color': 'red', 'linewidth': 2}
    )
    
    valid_data = merged_df[['GDP per capita, PPP (constant 2021 international $)', 'Researchers per Million']].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data['GDP per capita, PPP (constant 2021 international $)'], 
                        valid_data['Researchers per Million'])
        # axes[0, 0].text(0.05, 0.95, f'r = {r:.3f}', 
        #                  transform=axes[0, 0].transAxes, fontsize=11,
        #                  verticalalignment='top',
        #                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[0, 0].set_title('GDP per Capita → Researchers', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('GDP per capita (PPP)')
    axes[0, 0].set_ylabel('Researchers per Million')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: R&D Expenditure vs Researchers
    sns.regplot(
        data=merged_df,
        x='Mean R&D Expenditure (% GDP)',
        y='Researchers per Million',
        ax=axes[0, 1],
        scatter_kws={**scatter_kws, 'color': 'purple'},
        line_kws={'color': 'darkviolet', 'linewidth': 2}
    )
    
    valid_data = merged_df[['Mean R&D Expenditure (% GDP)', 'Researchers per Million']].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data['Mean R&D Expenditure (% GDP)'], 
                        valid_data['Researchers per Million'])
        # axes[0, 1].text(0.05, 0.95, f'r = {r:.3f}', 
        #                  transform=axes[0, 1].transAxes, fontsize=11,
        #                  verticalalignment='top',
        #                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[0, 1].set_title('R&D Expenditure → Researchers', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Mean R&D Expenditure (past 5 years, % GDP)')
    axes[0, 1].set_ylabel('Researchers per Million')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Education Spending vs Researchers
    sns.regplot(
        data=merged_df,
        x='Mean Education Spending (% GDP)',
        y='Researchers per Million',
        ax=axes[0, 2],
        scatter_kws={**scatter_kws, 'color': 'orange'},
        line_kws={'color': 'darkorange', 'linewidth': 2}
    )
    
    valid_data = merged_df[['Mean Education Spending (% GDP)', 'Researchers per Million']].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data['Mean Education Spending (% GDP)'], 
                        valid_data['Researchers per Million'])
        # axes[0, 2].text(0.05, 0.95, f'r = {r:.3f}', 
        #                  transform=axes[0, 2].transAxes, fontsize=11,
        #                  verticalalignment='top',
        #                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[0, 2].set_title('Education Spending → Researchers', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Mean Education Spending (past 5 years, % GDP)')
    axes[0, 2].set_ylabel('Researchers per Million')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Academic Freedom vs Researchers
    sns.regplot(
        data=merged_df,
        x='Academic Freedom Index',
        y='Researchers per Million',
        ax=axes[1, 0],
        scatter_kws={**scatter_kws, 'color': 'blue'},
        line_kws={'color': 'darkblue', 'linewidth': 2}
    )
    
    valid_data = merged_df[['Academic Freedom Index', 'Researchers per Million']].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data['Academic Freedom Index'], 
                        valid_data['Researchers per Million'])
        # axes[1, 0].text(0.05, 0.95, f'r = {r:.3f}', 
        #                  transform=axes[1, 0].transAxes, fontsize=11,
        #                  verticalalignment='top',
        #                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[1, 0].set_title('Academic Freedom → Researchers', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Academic Freedom Index')
    axes[1, 0].set_ylabel('Researchers per Million')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Population vs Researchers (log scale)
    merged_df_copy = merged_df.copy()
    merged_df_copy['log_pop'] = np.log10(merged_df_copy['Population'])
    
    sns.regplot(
        data=merged_df_copy,
        x='log_pop',
        y='Researchers per Million',
        ax=axes[1, 1],
        scatter_kws={**scatter_kws, 'color': 'green'},
        line_kws={'color': 'darkgreen', 'linewidth': 2}
    )
    
    valid_data = merged_df_copy[['log_pop', 'Researchers per Million']].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data['log_pop'], 
                        valid_data['Researchers per Million'])
        # axes[1, 1].text(0.05, 0.95, f'r = {r:.3f}', 
        #                  transform=axes[1, 1].transAxes, fontsize=11,
        #                  verticalalignment='top',
        #                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[1, 1].set_title('Population → Researchers (log scale)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Log10(Population)')
    axes[1, 1].set_ylabel('Researchers per Million')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Post-Soviet vs Non Post-Soviet comparison
    post_soviet_data = merged_df.copy()
    post_soviet_data['Country Type'] = post_soviet_data['Is Post-Soviet'].map(
        {0: 'Non Post-Soviet', 1: 'Post-Soviet'}
    )
    
    # Create violin plot
    sns.violinplot(
        data=post_soviet_data,
        x='Country Type',
        y='Researchers per Million',
        hue='Country Type',
        ax=axes[1, 2],
        palette=['lightblue', 'salmon'],
        legend=False
    )
    
    # Add mean lines
    ps_mean = merged_df[merged_df['Is Post-Soviet'] == 1]['Researchers per Million'].mean()
    non_ps_mean = merged_df[merged_df['Is Post-Soviet'] == 0]['Researchers per Million'].mean()
    
    axes[1, 2].axhline(ps_mean, xmin=0.5, xmax=1.0, color='darkred', linestyle='--', linewidth=2, label=f'PS Mean: {ps_mean:.0f}')
    axes[1, 2].axhline(non_ps_mean, xmin=0.0, xmax=0.5, color='darkblue', linestyle='--', linewidth=2, label=f'Non-PS Mean: {non_ps_mean:.0f}')
    
    axes[1, 2].set_title('Post-Soviet vs Non Post-Soviet', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Researchers per Million')
    axes[1, 2].set_xlabel('')
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotation
    # diff = ps_mean - non_ps_mean
    # axes[1, 2].text(0.5, 0.95, f'Difference: {diff:+.0f} researchers/million',
    #                  transform=axes[1, 2].transAxes, ha='right', va='top',
    #                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    #                  fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "global_relationships.png" if save else None
    _save_and_show(fig_path)


def plot_correlation_analysis(merged_df: pd.DataFrame, save: bool = True) -> None:
    """Correlation heatmap for overall dataset."""
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Select numeric columns
    numeric_cols = [
        'Researchers per Million',
        'GDP per capita, PPP (constant 2021 international $)',
        'Mean R&D Expenditure (% GDP)',
        'Mean Education Spending (% GDP)',
        'Academic Freedom Index',
        'Population',
        'Is Post-Soviet',
    ]
    
    # Compute correlation
    correlation_matrix = merged_df[numeric_cols].corr()
    
    # Create concise labels
    label_mapping = {
        'Researchers per Million': 'Researchers\n(Target)',
        'GDP per capita, PPP (constant 2021 international $)': 'GDP per capita\n(current year)',
        'Mean R&D Expenditure (% GDP)': 'R&D Spending\n(past 5 years)',
        'Mean Education Spending (% GDP)': 'Education\nSpending\n(past 5 years)',
        'Academic Freedom Index': 'Academic\nFreedom\n(current year)',
        'Population': 'Population\n(current year)',
        'Is Post-Soviet': 'Post-Soviet\nFlag',
    }
    
    correlation_matrix = correlation_matrix.rename(columns=label_mapping, index=label_mapping)
    
    # Plot 1: Full correlation heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1.5,
        cbar_kws={'shrink': 0.8},
        ax=axes[0],
        vmin=-1,
        vmax=1
    )
    axes[0].set_title('Overall Correlation Matrix\n(All Observations)', 
                      fontsize=14, fontweight='bold', pad=20)
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', rotation=0, labelsize=10)
    
    # Plot 2: Correlation with target only
    target_corr = correlation_matrix['Researchers\n(Target)'].drop('Researchers\n(Target)').sort_values(ascending=False)
    colors = ['darkgreen' if x > 0.5 else 'green' if x > 0 else 'red' for x in target_corr.values]
    
    axes[1].barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_yticks(range(len(target_corr)))
    axes[1].set_yticklabels(target_corr.index, fontsize=10)
    axes[1].axvline(x=0, color='black', linewidth=1.5)
    axes[1].set_xlabel('Correlation with Target', fontsize=11, fontweight='bold')
    axes[1].set_title('Predictor Strength Ranking', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xlim(-0.2, 1.0)
    
    # Add value labels
    for i, v in enumerate(target_corr.values):
        axes[1].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=10)
    
    # Add interpretation zones
    axes[1].axvspan(0.7, 1.0, alpha=0.1, color='green', label='Strong (>0.7)')
    axes[1].axvspan(0.3, 0.7, alpha=0.1, color='yellow', label='Moderate (0.3-0.7)')
    axes[1].axvspan(-0.1, 0.3, alpha=0.1, color='gray', label='Weak (<0.3)')
    axes[1].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "global_correlations.png" if save else None
    _save_and_show(fig_path)


def plot_top_countries(merged_df: pd.DataFrame, save: bool = True) -> None:
    """Show top countries by average performance (2019-2023)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Calculate global statistics
    global_mean = merged_df['Researchers per Million'].mean()
    global_median = merged_df['Researchers per Million'].median()
    
    # Calculate average per country with std across years
    country_stats = (
        merged_df.groupby('Entity')['Researchers per Million']
        .agg(['mean', 'std', 'count'])
        .sort_values('mean', ascending=False)
    )
    
    # Plot 1: Top 25 countries overall
    top_15 = country_stats.head(15)
    y_pos = range(len(top_15))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_15)))
    
    axes[0, 0].barh(y_pos, top_15['mean'], xerr=top_15['std'], 
                     color=colors, alpha=0.8, capsize=3, ecolor='gray')
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(top_15.index, fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel('Researchers per Million (Mean ± Std, 2019-2023)', fontweight='bold')
    axes[0, 0].set_title('Top 15 Countries (Overall Average)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].axvline(global_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Mean: {global_mean:.0f}')
    axes[0, 0].axvline(global_median, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Median: {global_median:.0f}')
    axes[0, 0].legend(fontsize=9)
    
    # Calculate stats for Post-Soviet countries
    ps_country_stats = (
        merged_df[merged_df['Is Post-Soviet'] == 1]
        .groupby('Entity')['Researchers per Million']
        .agg(['mean', 'std', 'count'])
        .sort_values('mean', ascending=False)
    )
    ps_top = ps_country_stats.head(15)
    
    # Plot 2: Post-Soviet top performers
    axes[0, 1].barh(range(len(ps_top)), ps_top['mean'], xerr=ps_top['std'],
                     color='salmon', alpha=0.8, capsize=3, ecolor='darkred')
    axes[0, 1].set_yticks(range(len(ps_top)))
    axes[0, 1].set_yticklabels(ps_top.index, fontsize=8)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('Researchers per Million (Mean ± Std, 2019-2023)', fontweight='bold')
    axes[0, 1].set_title('Top 15 Post-Soviet Countries', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    axes[0, 1].axvline(global_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Mean: {global_mean:.0f}')
    axes[0, 1].axvline(global_median, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Median: {global_median:.0f}')
    axes[0, 1].legend(fontsize=9)
    
    # Calculate stats for Non Post-Soviet countries
    non_ps_country_stats = (
        merged_df[merged_df['Is Post-Soviet'] == 0]
        .groupby('Entity')['Researchers per Million']
        .agg(['mean', 'std', 'count'])
        .sort_values('mean', ascending=False)
    )
    non_ps_top = non_ps_country_stats.head(15)
    
    # Plot 3: Non Post-Soviet
    axes[1, 0].barh(range(len(non_ps_top)), non_ps_top['mean'], xerr=non_ps_top['std'],
                     color='lightblue', alpha=0.8, capsize=3, ecolor='darkblue')
    axes[1, 0].set_yticks(range(len(non_ps_top)))
    axes[1, 0].set_yticklabels(non_ps_top.index, fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel('Researchers per Million (Mean ± Std, 2019-2023)', fontweight='bold')
    axes[1, 0].set_title('Non Post-Soviet Countries', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].axvline(global_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Mean: {global_mean:.0f}')
    axes[1, 0].axvline(global_median, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Median: {global_median:.0f}')
    axes[1, 0].legend(fontsize=9)
    
    # Plot 4: Distribution comparison
    ps_data = merged_df[merged_df['Is Post-Soviet'] == 1]['Researchers per Million']
    non_ps_data = merged_df[merged_df['Is Post-Soviet'] == 0]['Researchers per Million']
    
    axes[1, 1].hist([non_ps_data, ps_data], bins=25, label=['Non Post-Soviet', 'Post-Soviet'],
                     color=['lightblue', 'salmon'], alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(non_ps_data.mean(), color='blue', linestyle='--', linewidth=2, label=f'Non PS Mean: {non_ps_data.mean():.0f}')
    axes[1, 1].axvline(ps_data.mean(), color='red', linestyle='--', linewidth=2, label=f'PS Mean: {ps_data.mean():.0f}')
    axes[1, 1].axvline(global_mean, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Mean: {global_mean:.0f}')
    axes[1, 1].axvline(global_median, color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Global Median: {global_median:.0f}')
    axes[1, 1].set_xlabel('Researchers per Million', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Overall Distribution Comparison (2019-2023)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "global_top_countries.png" if save else None
    _save_and_show(fig_path)


def run_augmented_eda(merged_df: pd.DataFrame, save: bool = True) -> None:
    """
    Run comprehensive global EDA for augmented dataset.
    Focuses on overall patterns, not year-specific analysis.
    
    Args:
        merged_df: Augmented dataset with multiple years
        save: Whether to save figures
    """
    
    print("\nGenerating global visualizations (all years combined)...")
    print("="*80)
    
    print("1. Overall distributions and summary statistics...")
    plot_overall_distributions(merged_df, save=save)
    
    print("2. Global feature relationships...")
    plot_feature_relationships(merged_df, save=save)
    
    print("3. Correlation analysis...")
    plot_correlation_analysis(merged_df, save=save)
    
    print("4. Top countries ranking...")
    plot_top_countries(merged_df, save=save)
    
    print("="*80)
    print("✅ Global EDA complete! 4 comprehensive visualizations generated.")
    print(f"   Focus: Overall patterns across all {len(merged_df)} observations")
    print(f"   Period: {int(merged_df['Year'].min())}-{int(merged_df['Year'].max())}")
    print(f"   Saved to: {FIGURES_DIR}/")
    print("="*80)
