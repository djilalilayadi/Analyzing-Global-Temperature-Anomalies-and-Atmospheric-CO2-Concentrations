# data_handler.py

import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_co2_data(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Loads raw CO2 concentration data from a specified file path.

    Args:
        filepath (Path): Path to the CO2 data file.

    Returns:
        pd.DataFrame: Raw CO2 data or None if loading fails.
    """
    try:
        logger.info(f"Loading CO2 data from: {filepath}")
        co2_df = pd.read_csv(
            filepath,
            skiprows=42,  # Adjust if your file has a different header length
            sep=r'\s+',
            header=None,
            names=['year', 'month', 'decimal_date', 'monthly_average',
                   'detrended', 'n_days', 'st_dev', 'uncert'],
            na_values=[-99.99],
            dtype={
                'year': int, 'month': int, 'decimal_date': float,
                'monthly_average': float, 'detrended': float,
                'n_days': int, 'st_dev': float, 'uncert': float
            }
        )
        logger.info(f"CO2 data loaded successfully. Shape: {co2_df.shape}")
        return co2_df
    except FileNotFoundError:
        logger.error(f"CO2 file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading CO2 data: {e}")
        return None

def clean_co2_data(co2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares CO2 data for analysis.

    Args:
        co2_df (pd.DataFrame): Raw CO2 data.

    Returns:
        pd.DataFrame: Cleaned CO2 data with 'date' as index.
    """
    logger.info("Cleaning CO2 data...")
    co2_cleaned = co2_df[['year', 'month', 'monthly_average']].copy()
    co2_cleaned.dropna(subset=['monthly_average'], inplace=True)
    co2_cleaned['date'] = pd.to_datetime(
        co2_cleaned['year'].astype(str) + '-' + co2_cleaned['month'].astype(str),
        format='%Y-%m'
    )
    co2_cleaned.set_index('date', inplace=True)
    co2_cleaned.drop(columns=['year', 'month'], inplace=True)
    logger.info(f"CO2 data cleaned. Final shape: {co2_cleaned.shape}")
    return co2_cleaned

def load_temperature_data(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Loads raw temperature anomaly data from a specified file path.

    Args:
        filepath (Path): Path to the temperature data file.

    Returns:
        pd.DataFrame: Raw temperature data or None if loading fails.
    """
    try:
        logger.info(f"Loading temperature data from: {filepath}")
        temp_df = pd.read_csv(
            filepath,
            skiprows=1,
            header=0,
            na_values=['***', '****', '*****'],
            index_col=0
        )
        logger.info(f"Temperature data loaded successfully. Shape: {temp_df.shape}")
        return temp_df
    except FileNotFoundError:
        logger.error(f"Temperature file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading temperature data: {e}")
        return None

def clean_temperature_data(temp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and reshapes temperature data from wide to long format.

    Args:
        temp_df (pd.DataFrame): Raw temperature data.

    Returns:
        pd.DataFrame: Cleaned temperature data with 'date' as index.
    """
    logger.info("Cleaning temperature data...")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_cols = [col for col in temp_df.columns if col in month_names]

    temp_monthly = temp_df[monthly_cols].reset_index()
    temp_monthly.rename(columns={'Year': 'year'}, inplace=True)

    temp_melted = temp_monthly.melt(
        id_vars=['year'],
        var_name='month_name',
        value_name='temperature_anomaly'
    )

    month_to_num = {name: i+1 for i, name in enumerate(month_names)}
    temp_melted['month'] = temp_melted['month_name'].map(month_to_num)
    temp_melted.dropna(subset=['temperature_anomaly'], inplace=True)

    temp_melted['date'] = pd.to_datetime(
        temp_melted['year'].astype(str) + '-' + temp_melted['month'].astype(str),
        format='%Y-%m'
    )

    temp_cleaned = temp_melted.set_index('date')
    temp_cleaned = temp_cleaned[['temperature_anomaly']].copy()
    temp_cleaned['temperature_anomaly'] = pd.to_numeric(temp_cleaned['temperature_anomaly'])
    logger.info(f"Temperature data cleaned. Final shape: {temp_cleaned.shape}")
    return temp_cleaned

def merge_datasets(co2_df: pd.DataFrame, temp_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merges CO2 and temperature datasets on their date index.

    Args:
        co2_df (pd.DataFrame): Cleaned CO2 data.
        temp_df (pd.DataFrame): Cleaned temperature data.

    Returns:
        pd.DataFrame: Merged dataset or None if input DataFrames are invalid.
    """
    if co2_df is None or temp_df is None or co2_df.empty or temp_df.empty:
        logger.error("Cannot merge: one or both input DataFrames are invalid or empty.")
        return None

    logger.info("Merging datasets...")
    merged_df = pd.merge(
        co2_df,
        temp_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    logger.info(f"Datasets merged successfully. Shape: {merged_df.shape}")
    return merged_df

