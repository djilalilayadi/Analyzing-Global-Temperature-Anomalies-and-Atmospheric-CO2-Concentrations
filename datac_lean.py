import pandas as pd
import matplotlib.pyplot as plt # We'll use this later for plotting

# --- CO2 Data Loading ---
# Adjust 'your_co2_file.txt' to the actual path of your downloaded CO2 file
# The NOAA file usually has several lines of comments at the beginning,
# and the data columns are separated by spaces.
# The missing value indicator is often -99.99 or similar.

try:
    co2_df = pd.read_csv(
        'co2_mm_mlo.txt', # Replace with your file name/path
        skiprows=72,      # Based on typical NOAA Mauna Loa file structure (adjust if needed)
        sep=r'\s+',       # Use regex to split by one or more spaces
        header=None,      # No header row in the data part
        names=['year', 'month', 'decimal_date', 'monthly_average', 'detrended', 'n_days', 'st_dev', 'uncert'],
        na_values=[-99.99] # Specify missing value indicator
    )
    print("CO2 Data Loaded Successfully!")
    print(co2_df.head())
    print(co2_df.info())

except FileNotFoundError:
    print("Error: CO2 file not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred while loading CO2 data: {e}")

