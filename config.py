# config.py

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- File Paths ---
# Get the directory where this config.py file is located
_current_dir = Path(__file__).parent
DATA_DIR = _current_dir / 'data' # Assumes a 'data' subdirectory

CO2_FILE_PATH = DATA_DIR / 'co2_mm_mlo.txt'
TEMP_FILE_PATH = DATA_DIR / 'global_temp_anomalies.csv'

# --- Logging Configuration ---
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging():
    """Configures the basic logging for the application."""
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    # Optional: Suppress excessive logging from libraries if needed
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)

# --- Plotting Style Configuration ---
def apply_plot_style():
    """Applies a consistent plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

