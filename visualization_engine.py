# visualization_engine.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def create_time_series_plot(df: pd.DataFrame) -> None:
    """
    Creates a dual-axis time series plot showing CO2 concentration and temperature anomaly trends.

    Args:
        df (pd.DataFrame): Merged dataset containing 'monthly_average' and 'temperature_anomaly'.
    """
    if df is None or df.empty:
        logger.warning("DataFrame is empty or None. Cannot create time series plot.")
        return

    logger.info("Creating time series plot...")
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # CO2 concentration plot
    color_co2 = '#1f77b4'
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CO2 Concentration (ppm)', color=color_co2, fontsize=12, fontweight='bold')
    line1 = ax1.plot(df.index, df['monthly_average'],
                     color=color_co2, linewidth=2, label='CO2 Concentration', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_co2)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Temperature anomaly plot
    ax2 = ax1.twinx()
    color_temp = '#d62728'
    ax2.set_ylabel('Temperature Anomaly (°C)', color=color_temp, fontsize=12, fontweight='bold')
    line2 = ax2.plot(df.index, df['temperature_anomaly'],
                     color=color_temp, linewidth=2, label='Temperature Anomaly', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_temp)

    # Styling
    plt.title('Global CO2 Concentration vs Temperature Anomaly Over Time',
              fontsize=16, fontweight='bold', pad=20)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.show()
    plt.close(fig) # Close the figure to free memory

def create_scatter_plot(df: pd.DataFrame, regression_results: Dict[str, float]) -> None:
    """
    Creates a scatter plot of temperature anomaly vs. CO2 concentration with a regression line.

    Args:
        df (pd.DataFrame): Merged dataset containing 'monthly_average' and 'temperature_anomaly'.
        regression_results (Dict[str, float]): Dictionary of regression statistics.
    """
    if df is None or df.empty:
        logger.warning("DataFrame is empty or None. Cannot create scatter plot.")
        return

    logger.info("Creating scatter plot...")
    fig = plt.figure(figsize=(10, 8))

    # Scatter plot with regression line
    sns.regplot(
        x='monthly_average',
        y='temperature_anomaly',
        data=df,
        scatter_kws={'alpha': 0.6, 's': 15, 'color': '#2E86AB'},
        line_kws={'color': '#A23B72', 'linewidth': 3}
    )

    # Add regression equation and R² to plot
    equation_text = (f"y = {regression_results['slope']:.4f}x + {regression_results['intercept']:.4f}\n"
                     f"R² = {regression_results['r_squared']:.4f}")

    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, fontweight='bold')

    plt.title('Temperature Anomaly vs CO2 Concentration', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('CO2 Concentration (ppm)', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature Anomaly (°C)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig) # Close the figure to free memory

