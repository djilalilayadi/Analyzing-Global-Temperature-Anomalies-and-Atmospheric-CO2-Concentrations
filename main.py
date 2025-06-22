# main.py

"""
Climate Data Analysis: CO2 Concentration vs Temperature Anomaly
=============================================================

This script orchestrates the analysis of the relationship between atmospheric CO2 concentrations
and global temperature anomalies using modularized components for data handling, analysis,
visualization, and reporting.

Author: Climate Data Analyst
Date: 2025
"""

import logging
from typing import Optional

# Import modules from your project
import config
import data_handler
import analysis_engine
import visualization_engine
import reporting_engine

logger = logging.getLogger(__name__)

class ClimateDataAnalyzer:
    """
    A class to orchestrate the climate data analysis pipeline.
    """

    def __init__(self, co2_file_path: str, temp_file_path: str):
        """
        Initialize the analyzer with file paths.

        Args:
            co2_file_path (str): Path to CO2 data file.
            temp_file_path (str): Path to temperature data file.
        """
        self.co2_file_path = co2_file_path
        self.temp_file_path = temp_file_path
        self.co2_df = None
        self.temp_df = None
        self.merged_df = None
        self.correlation = None
        self.regression_results = None

    def run_analysis(self) -> None:
        """
        Executes the complete analysis pipeline:
        1. Loads and cleans data.
        2. Merges datasets.
        3. Calculates statistics.
        4. Creates visualizations.
        5. Prints summary reports.
        """
        logger.info("Starting climate data analysis pipeline...")

        # --- 1. Data Loading and Cleaning ---
        co2_raw = data_handler.load_co2_data(config.CO2_FILE_PATH)
        if co2_raw is None:
            logger.error("Failed to load CO2 data. Aborting analysis.")
            return
        self.co2_df = data_handler.clean_co2_data(co2_raw)

        temp_raw = data_handler.load_temperature_data(config.TEMP_FILE_PATH)
        if temp_raw is None:
            logger.error("Failed to load temperature data. Aborting analysis.")
            return
        self.temp_df = data_handler.clean_temperature_data(temp_raw)

        # --- 2. Merge Datasets ---
        self.merged_df = data_handler.merge_datasets(self.co2_df, self.temp_df)
        if self.merged_df is None:
            logger.error("Failed to merge datasets. Aborting analysis.")
            return

        # --- 3. Calculate Statistics ---
        self.correlation, self.regression_results = analysis_engine.calculate_statistics(self.merged_df)

        # --- 4. Create Visualizations ---
        logger.info("Generating plots...")
        visualization_engine.create_time_series_plot(self.merged_df)
        visualization_engine.create_scatter_plot(self.merged_df, self.regression_results)

        # --- 5. Print Summary Report ---
        reporting_engine.print_summary_statistics(self.merged_df, self.correlation, self.regression_results)

        logger.info("Climate data analysis pipeline completed successfully!")


def main():
    """
    Main function to set up and execute the climate data analysis.
    """
    # Setup logging and plotting style
    config.setup_logging()
    config.apply_plot_style()

    # Create analyzer instance and run analysis
    # File paths are now managed in config.py
    analyzer = ClimateDataAnalyzer(str(config.CO2_FILE_PATH), str(config.TEMP_FILE_PATH))
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

