import pandas as pd
import numpy as np
import requests
import zipfile
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from typing import Tuple, Dict, Any
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader and preprocessor for household energy consumption data."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config['data']
        self.preprocessing_config = self.config['preprocessing']
        self.scaler = None

    def download_data(self) -> None:
        """Download household power consumption dataset from UCI repository."""
        url = self.data_config['dataset_url']
        raw_data_dir = os.path.dirname(self.data_config['raw_data_path'])

        # Create directory if it doesn't exist
        os.makedirs(raw_data_dir, exist_ok=True)

        # Download the zip file
        zip_path = os.path.join(raw_data_dir, "household_power_consumption.zip")

        if not os.path.exists(zip_path):
            logger.info(f"Downloading data from {url}")
            response = requests.get(url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)

        # Extract the zip file
        if not os.path.exists(self.data_config['raw_data_path']):
            logger.info("Extracting data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw household power consumption data."""
        # Download data if it doesn't exist
        if not os.path.exists(self.data_config['raw_data_path']):
            self.download_data()

        logger.info("Loading raw data...")

        # Load the data
        df = pd.read_csv(
            self.data_config['raw_data_path'],
            sep=';',
            parse_dates={'datetime': ['Date', 'Time']},
            index_col='datetime',
            na_values=['?'],
            low_memory=False
        )

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        strategy = self.preprocessing_config['missing_value_strategy']

        logger.info(f"Handling missing values using strategy: {strategy}")

        if strategy == "forward_fill":
            df = df.fillna(method='ffill')
        elif strategy == "interpolate":
            df = df.interpolate(method='time')
        elif strategy == "drop":
            df = df.dropna()

        # Fill any remaining NaN values with 0
        df = df.fillna(0)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataset."""
        logger.info("Adding time features...")

        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def add_lag_features(self, df: pd.DataFrame, target_col: str = 'Global_active_power') -> pd.DataFrame:
        """Add lag features for time series."""
        lag_periods = self.preprocessing_config['feature_engineering']['lag_periods']

        logger.info(f"Adding lag features for periods: {lag_periods}")

        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        return df

    def add_rolling_features(self, df: pd.DataFrame, target_col: str = 'Global_active_power') -> pd.DataFrame:
        """Add rolling window features."""
        windows = self.preprocessing_config['feature_engineering']['rolling_windows']

        logger.info(f"Adding rolling features for windows: {windows}")

        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()

        return df

    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale features using the specified scaling method."""
        scaling_method = self.preprocessing_config['scaling_method']

        logger.info(f"Scaling features using method: {scaling_method}")

        if scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()

        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def preprocess_data(self) -> Dict[str, Any]:
        """Complete data preprocessing pipeline."""
        logger.info("Starting data preprocessing pipeline...")

        # Load raw data
        df = self.load_raw_data()

        # Handle missing values
        df = self.handle_missing_values(df)

        # Feature engineering
        if self.preprocessing_config['feature_engineering']['add_time_features']:
            df = self.add_time_features(df)

        if self.preprocessing_config['feature_engineering']['add_lag_features']:
            df = self.add_lag_features(df)

        if self.preprocessing_config['feature_engineering']['rolling_window_features']:
            df = self.add_rolling_features(df)

        # Remove rows with NaN values (due to lag and rolling features)
        df = df.dropna()

        # Define target variable
        target_col = 'Global_active_power'
        feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols]
        y = df[target_col]

        # Split data chronologically
        train_size = int(len(df) * self.data_config['train_split'])
        val_size = int(len(df) * self.data_config['validation_split'])

        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]

        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]

        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)

        # Save processed data
        processed_dir = self.data_config['processed_data_path']
        os.makedirs(processed_dir, exist_ok=True)

        X_train_scaled.to_csv(os.path.join(processed_dir, 'X_train.csv'))
        X_val_scaled.to_csv(os.path.join(processed_dir, 'X_val.csv'))
        X_test_scaled.to_csv(os.path.join(processed_dir, 'X_test.csv'))
        y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'))
        y_val.to_csv(os.path.join(processed_dir, 'y_val.csv'))
        y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'))

        logger.info("Data preprocessing completed successfully!")

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': self.scaler
        }

    def load_processed_data(self) -> Dict[str, Any]:
        """Load preprocessed data from files."""
        processed_dir = self.data_config['processed_data_path']

        # Check if processed data exists
        files_to_check = ['X_train.csv', 'X_val.csv', 'X_test.csv', 'y_train.csv', 'y_val.csv', 'y_test.csv']
        if not all(os.path.exists(os.path.join(processed_dir, f)) for f in files_to_check):
            logger.info("Processed data not found. Running preprocessing...")
            return self.preprocess_data()

        logger.info("Loading processed data from files...")

        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'), index_col=0)
        X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'), index_col=0)
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'), index_col=0).squeeze()
        y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv'), index_col=0).squeeze()
        y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'), index_col=0).squeeze()

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    data = loader.preprocess_data()
    print("Data preprocessing completed!")
    print(f"Training data shape: {data['X_train'].shape}")
    print(f"Validation data shape: {data['X_val'].shape}")
    print(f"Test data shape: {data['X_test'].shape}")