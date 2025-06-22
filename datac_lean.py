"""
Climate Data Analysis: CO2 Concentration vs Temperature Anomaly
=============================================================

This script analyzes the relationship between atmospheric CO2 concentrations
and global temperature anomalies using data visualization and statistical analysis.

Author: Climate Data Analyst
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib and seaborn styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ClimateDataAnalyzer:
    """
    A class to handle climate data loading, processing, and analysis.
    """
    
    def __init__(self, co2_file_path: str, temp_file_path: str):
        """
        Initialize the analyzer with file paths.
        
        Args:
            co2_file_path (str): Path to CO2 data file
            temp_file_path (str): Path to temperature data file
        """
        self.co2_file_path = Path(co2_file_path)
        self.temp_file_path = Path(temp_file_path)
        self.co2_df = None
        self.temp_df = None
        self.merged_df = None
        
    def load_co2_data(self) -> Optional[pd.DataFrame]:
        """
        Load and process CO2 concentration data.
        
        Returns:
            pd.DataFrame: Processed CO2 data or None if loading fails
        """
        try:
            logger.info("Loading CO2 data...")
            
            co2_df = pd.read_csv(
                self.co2_file_path,
                skiprows=42,
                sep=r'\s+',
                header=None,
                names=['year', 'month', 'decimal_date', 'monthly_average', 
                      'detrended', 'n_days', 'st_dev', 'uncert'],
                na_values=[-99.99],
                dtype={
                    'year': int,
                    'month': int,
                    'decimal_date': float,
                    'monthly_average': float,
                    'detrended': float,
                    'n_days': int,
                    'st_dev': float,
                    'uncert': float
                }
            )
            
            logger.info(f"CO2 data loaded successfully. Shape: {co2_df.shape}")
            return co2_df
            
        except FileNotFoundError:
            logger.error(f"CO2 file not found: {self.co2_file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading CO2 data: {e}")
            return None
    
    def clean_co2_data(self, co2_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare CO2 data for analysis.
        
        Args:
            co2_df (pd.DataFrame): Raw CO2 data
            
        Returns:
            pd.DataFrame: Cleaned CO2 data
        """
        logger.info("Cleaning CO2 data...")
        
        # Select relevant columns and remove missing values
        co2_cleaned = co2_df[['year', 'month', 'monthly_average']].copy()
        co2_cleaned.dropna(subset=['monthly_average'], inplace=True)
        
        # Create datetime index
        co2_cleaned['date'] = pd.to_datetime(
            co2_cleaned['year'].astype(str) + '-' + co2_cleaned['month'].astype(str),
            format='%Y-%m'
        )
        co2_cleaned.set_index('date', inplace=True)
        co2_cleaned.drop(columns=['year', 'month'], inplace=True)
        
        logger.info(f"CO2 data cleaned. Final shape: {co2_cleaned.shape}")
        return co2_cleaned
    
    def load_temperature_data(self) -> Optional[pd.DataFrame]:
        """
        Load temperature anomaly data.
        
        Returns:
            pd.DataFrame: Raw temperature data or None if loading fails
        """
        try:
            logger.info("Loading temperature data...")
            
            temp_df = pd.read_csv(
                self.temp_file_path,
                skiprows=1,
                header=0,
                na_values=['***', '****', '*****'],
                index_col=0
            )
            
            logger.info(f"Temperature data loaded successfully. Shape: {temp_df.shape}")
            return temp_df
            
        except FileNotFoundError:
            logger.error(f"Temperature file not found: {self.temp_file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading temperature data: {e}")
            return None
    
    def clean_temperature_data(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and reshape temperature data from wide to long format.
        
        Args:
            temp_df (pd.DataFrame): Raw temperature data
            
        Returns:
            pd.DataFrame: Cleaned temperature data
        """
        logger.info("Cleaning temperature data...")
        
        # Define month columns
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_cols = [col for col in temp_df.columns if col in month_names]
        
        # Reshape from wide to long format
        temp_monthly = temp_df[monthly_cols].reset_index()
        temp_monthly.rename(columns={'Year': 'year'}, inplace=True)
        
        temp_melted = temp_monthly.melt(
            id_vars=['year'],
            var_name='month_name',
            value_name='temperature_anomaly'
        )
        
        # Convert month names to numbers
        month_to_num = {name: i+1 for i, name in enumerate(month_names)}
        temp_melted['month'] = temp_melted['month_name'].map(month_to_num)
        temp_melted.dropna(subset=['temperature_anomaly'], inplace=True)
        
        # Create datetime index
        temp_melted['date'] = pd.to_datetime(
            temp_melted['year'].astype(str) + '-' + temp_melted['month'].astype(str),
            format='%Y-%m'
        )
        
        temp_cleaned = temp_melted.set_index('date')
        temp_cleaned = temp_cleaned[['temperature_anomaly']].copy()
        temp_cleaned['temperature_anomaly'] = pd.to_numeric(temp_cleaned['temperature_anomaly'])
        
        logger.info(f"Temperature data cleaned. Final shape: {temp_cleaned.shape}")
        return temp_cleaned
    
    def merge_datasets(self, co2_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge CO2 and temperature datasets on date index.
        
        Args:
            co2_df (pd.DataFrame): Cleaned CO2 data
            temp_df (pd.DataFrame): Cleaned temperature data
            
        Returns:
            pd.DataFrame: Merged dataset
        """
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
    
    def calculate_statistics(self, merged_df: pd.DataFrame) -> Tuple[float, dict]:
        """
        Calculate correlation and linear regression statistics.
        
        Args:
            merged_df (pd.DataFrame): Merged dataset
            
        Returns:
            Tuple[float, dict]: Correlation coefficient and regression results
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
        
        return correlation, regression_results
    
    def create_time_series_plot(self, merged_df: pd.DataFrame) -> None:
        """
        Create a dual-axis time series plot showing CO2 and temperature trends.
        
        Args:
            merged_df (pd.DataFrame): Merged dataset
        """
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # CO2 concentration plot
        color_co2 = '#1f77b4'
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('CO2 Concentration (ppm)', color=color_co2, fontsize=12, fontweight='bold')
        line1 = ax1.plot(merged_df.index, merged_df['monthly_average'], 
                        color=color_co2, linewidth=2, label='CO2 Concentration', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color_co2)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Temperature anomaly plot
        ax2 = ax1.twinx()
        color_temp = '#d62728'
        ax2.set_ylabel('Temperature Anomaly (°C)', color=color_temp, fontsize=12, fontweight='bold')
        line2 = ax2.plot(merged_df.index, merged_df['temperature_anomaly'], 
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
    
    def create_scatter_plot(self, merged_df: pd.DataFrame, regression_results: dict) -> None:
        """
        Create a scatter plot with regression line.
        
        Args:
            merged_df (pd.DataFrame): Merged dataset
            regression_results (dict): Regression statistics
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot with regression line
        sns.regplot(
            x='monthly_average',
            y='temperature_anomaly',
            data=merged_df,
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
    
    def print_summary_statistics(self, merged_df: pd.DataFrame, correlation: float, 
                               regression_results: dict) -> None:
        """
        Print comprehensive summary statistics.
        
        Args:
            merged_df (pd.DataFrame): Merged dataset
            correlation (float): Correlation coefficient
            regression_results (dict): Regression statistics
        """
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
    
    def run_analysis(self) -> None:
        """
        Execute the complete analysis pipeline.
        """
        logger.info("Starting climate data analysis...")
        
        # Load and clean CO2 data
        co2_raw = self.load_co2_data()
        if co2_raw is None:
            logger.error("Failed to load CO2 data. Exiting.")
            return
        
        self.co2_df = self.clean_co2_data(co2_raw)
        
        # Load and clean temperature data
        temp_raw = self.load_temperature_data()
        if temp_raw is None:
            logger.error("Failed to load temperature data. Exiting.")
            return
        
        self.temp_df = self.clean_temperature_data(temp_raw)
        
        # Merge datasets
        self.merged_df = self.merge_datasets(self.co2_df, self.temp_df)
        
        # Calculate statistics
        correlation, regression_results = self.calculate_statistics(self.merged_df)
        
        # Create visualizations
        self.create_time_series_plot(self.merged_df)
        self.create_scatter_plot(self.merged_df, regression_results)
        
        # Print summary
        self.print_summary_statistics(self.merged_df, correlation, regression_results)
        
        logger.info("Analysis completed successfully!")


def main():
    """
    Main function to execute the climate data analysis.
    """
    # File paths (update these to match your file locations)
    CO2_FILE_PATH = r'C:\Users\layad\OneDrive\سطح المكتب\1-lessons\projects\Data co2\data\co2_mm_mlo.txt'
    TEMP_FILE_PATH = r'C:\Users\layad\OneDrive\سطح المكتب\1-lessons\projects\Data co2\data\global_temp_anomalies.csv'
    
    # Create analyzer instance and run analysis
    analyzer = ClimateDataAnalyzer(CO2_FILE_PATH, TEMP_FILE_PATH)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
