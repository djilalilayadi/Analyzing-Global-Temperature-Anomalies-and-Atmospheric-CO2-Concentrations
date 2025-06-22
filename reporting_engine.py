# reporting_engine.py

import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def print_summary_statistics(merged_df: pd.DataFrame, correlation: float,
                             regression_results: Dict[str, float]) -> None:
    """
    Prints comprehensive summary statistics of the analysis.

    Args:
        merged_df (pd.DataFrame): Merged dataset.
        correlation (float): Pearson correlation coefficient.
        regression_results (Dict[str, float]): Dictionary of regression statistics.
    """
    logger.info("Printing summary statistics...")

    print("\n" + "="*60)
    print("CLIMATE DATA ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nDataset Overview:")
    print(f"  • Time period: {merged_df.index.min().strftime('%Y-%m')} to {merged_df.index.max().strftime('%Y-%m')}")
    print(f"  • Total observations: {len(merged_df):,}")
    print(f"  • Duration: {(merged_df.index.max() - merged_df.index.min()).days / 365.25:.1f} years")

    print(f"\nDescriptive Statistics:")
    print(merged_df.describe().round(4))

    print(f"\nCorrelation Analysis:")
    print(f"  • Pearson correlation: {correlation:.4f}")
    print(f"  • Strength: {'Very Strong' if abs(correlation) > 0.8 else 'Strong' if abs(correlation) > 0.6 else 'Moderate'}")

    print(f"\nLinear Regression Results:")
    print(f"  • Slope: {regression_results['slope']:.6f} (°C per ppm CO2)")
    print(f"  • Intercept: {regression_results['intercept']:.4f}")
    print(f"  • R-squared: {regression_results['r_squared']:.4f}")
    print(f"  • P-value: {regression_results['p_value']:.2e}")
    print(f"  • Standard error: {regression_results['std_err']:.6f}")

    # Interpretation
    temp_increase_per_100ppm = regression_results['slope'] * 100
    print(f"\nInterpretation:")
    print(f"  • For every 100 ppm increase in CO2, global temperature anomaly")
    print(f"    increases by approximately {temp_increase_per_100ppm:.2f}°C")
    print(f"  • The model explains {regression_results['r_squared']*100:.1f}% of temperature variance")
    if regression_results['p_value'] < 0.05:
        print(f"  • The very low p-value ({regression_results['p_value']:.2e}) indicates that the observed relationship is statistically significant and unlikely to have occurred by random chance.")
    else:
        print(f"  • The p-value ({regression_results['p_value']:.2e}) suggests the relationship might not be statistically significant.")

    print("\n" + "="*60)
    logger.info("Summary statistics printed.")

