# analysis_engine.py

import pandas as pd
from scipy.stats import linregress
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

def calculate_statistics(merged_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Calculates correlation and linear regression statistics between CO2 and temperature.

    Args:
        merged_df (pd.DataFrame): Merged dataset containing 'monthly_average' and 'temperature_anomaly'.

    Returns:
        Tuple[float, Dict[str, float]]: Correlation coefficient and a dictionary
                                         of regression results.
    """
    logger.info("Calculating statistics...")

    # Calculate correlation
    correlation = merged_df['monthly_average'].corr(merged_df['temperature_anomaly'])

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        merged_df['monthly_average'],
        merged_df['temperature_anomaly']
    )

    regression_results = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

    logger.info("Statistics calculated.")
    return correlation, regression_results

