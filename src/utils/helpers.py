import time
"""
Utility functions for the household energy consumption prediction project.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml"):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def ensure_dir(directory: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)

def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """Calculate memory usage of a DataFrame."""
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    return {
        'total_memory_mb': f"{memory_mb:.2f} MB",
        'memory_per_row': f"{memory_mb / len(df) * 1024:.2f} KB"
    }

def detect_outliers(series: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    """Detect outliers in a pandas Series."""
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
    else:
        raise ValueError(f"Unknown method: {method}")

    return outliers

def plot_style_config():
    """Configure matplotlib and seaborn plot styles."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def create_time_features(df: pd.DataFrame, datetime_col: str = None) -> pd.DataFrame:
    """Create time-based features from datetime index or column."""
    df_copy = df.copy()

    if datetime_col:
        dt = pd.to_datetime(df_copy[datetime_col])
    else:
        dt = df_copy.index

    df_copy['year'] = dt.year
    df_copy['month'] = dt.month
    df_copy['day'] = dt.day
    df_copy['hour'] = dt.hour
    df_copy['dayofweek'] = dt.dayofweek
    df_copy['dayofyear'] = dt.dayofyear
    df_copy['quarter'] = dt.quarter
    df_copy['weekofyear'] = dt.isocalendar().week

    # Cyclical encoding
    df_copy['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
    df_copy['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
    df_copy['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
    df_copy['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * dt.month / 12)

    return df_copy

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return float('inf') if new_value > 0 else 0
    return ((new_value - old_value) / old_value) * 100

def validate_data_split(train_size: float, val_size: float, test_size: float) -> bool:
    """Validate that data split proportions sum to 1."""
    total = train_size + val_size + test_size
    return abs(total - 1.0) < 1e-6

def print_section(title: str, char: str = "=", width: int = 50):
    """Print a formatted section header."""
    print()
    print(char * width)
    print(f" {title} ".center(width))
    print(char * width)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

class Timer:
    """Simple timer context manager."""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"{self.description} completed in {format_duration(elapsed_time)}")

# Data validation functions
def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """Validate DataFrame and return validation report."""
    validation_report = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': memory_usage(df),
        'is_valid': True,
        'errors': []
    }

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Missing required columns: {missing_cols}")

    # Check for empty DataFrame
    if df.empty:
        validation_report['is_valid'] = False
        validation_report['errors'].append("DataFrame is empty")

    return validation_report

def log_model_performance(model_name: str, metrics: Dict[str, float], logger: logging.Logger = None):
    """Log model performance metrics."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Model Performance - {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

# Export functions for easy importing
__all__ = [
    'setup_logging', 'load_config', 'save_config', 'ensure_dir', 'get_project_root',
    'format_duration', 'memory_usage', 'detect_outliers', 'plot_style_config',
    'create_time_features', 'calculate_percentage_change', 'validate_data_split',
    'print_section', 'safe_divide', 'Timer', 'validate_dataframe', 'log_model_performance'
]