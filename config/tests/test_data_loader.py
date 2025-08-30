"""
Unit tests for the data loader module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_processing.data_loader import DataLoader
from src.utils.helpers import validate_dataframe, detect_outliers

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()

        # Create sample test data
        dates = pd.date_range('2020-01-01', periods=1000, freq='H')
        self.sample_data = pd.DataFrame({
            'Global_active_power': np.random.normal(2.0, 0.5, 1000),
            'Global_reactive_power': np.random.normal(0.2, 0.1, 1000),
            'Voltage': np.random.normal(240, 5, 1000),
            'Global_intensity': np.random.normal(8.5, 2, 1000),
            'Sub_metering_1': np.random.poisson(1, 1000),
            'Sub_metering_2': np.random.poisson(1, 1000),
            'Sub_metering_3': np.random.poisson(2, 1000)
        }, index=dates)

    def test_handle_missing_values_forward_fill(self):
        """Test forward fill missing value strategy."""
        # Add some missing values
        test_data = self.sample_data.copy()
        test_data.iloc[10:15, 0] = np.nan

        # Set strategy to forward fill
        self.data_loader.preprocessing_config['missing_value_strategy'] = 'forward_fill'

        # Handle missing values
        result = self.data_loader.handle_missing_values(test_data)

        # Check that no NaN values remain
        self.assertFalse(result.isnull().any().any())

    def test_add_time_features(self):
        """Test time feature engineering."""
        result = self.data_loader.add_time_features(self.sample_data)

        # Check that time features were added
        expected_features = ['hour', 'day_of_week', 'month', 'quarter', 
                           'day_of_year', 'week_of_year',
                           'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                           'month_sin', 'month_cos']

        for feature in expected_features:
            self.assertIn(feature, result.columns)

        # Check that hour values are in correct range
        self.assertTrue(result['hour'].min() >= 0)
        self.assertTrue(result['hour'].max() <= 23)

        # Check that cyclical features are in correct range
        self.assertTrue(result['hour_sin'].min() >= -1)
        self.assertTrue(result['hour_sin'].max() <= 1)
        self.assertTrue(result['hour_cos'].min() >= -1)  
        self.assertTrue(result['hour_cos'].max() <= 1)

    def test_add_lag_features(self):
        """Test lag feature creation."""
        # Set lag periods
        self.data_loader.preprocessing_config['feature_engineering']['lag_periods'] = [1, 7, 24]

        result = self.data_loader.add_lag_features(self.sample_data, 'Global_active_power')

        # Check that lag features were added
        expected_lag_features = ['Global_active_power_lag_1', 
                               'Global_active_power_lag_7', 
                               'Global_active_power_lag_24']

        for feature in expected_lag_features:
            self.assertIn(feature, result.columns)

    def test_add_rolling_features(self):
        """Test rolling window feature creation."""
        # Set rolling windows
        self.data_loader.preprocessing_config['feature_engineering']['rolling_windows'] = [7, 24]

        result = self.data_loader.add_rolling_features(self.sample_data, 'Global_active_power')

        # Check that rolling features were added
        expected_features = [
            'Global_active_power_rolling_mean_7',
            'Global_active_power_rolling_std_7',
            'Global_active_power_rolling_min_7',
            'Global_active_power_rolling_max_7',
            'Global_active_power_rolling_mean_24',
            'Global_active_power_rolling_std_24',
            'Global_active_power_rolling_min_24',
            'Global_active_power_rolling_max_24'
        ]

        for feature in expected_features:
            self.assertIn(feature, result.columns)

class TestHelpers(unittest.TestCase):
    """Test cases for helper functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_series = pd.Series([1, 2, 3, 100, 4, 5, 6])  # 100 is outlier
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        outliers = detect_outliers(self.sample_series, method='iqr')

        # Should detect the value 100 as an outlier
        self.assertTrue(outliers.iloc[3])  # 100 should be detected as outlier
        self.assertFalse(outliers.iloc[0])  # 1 should not be an outlier

    def test_detect_outliers_zscore(self):
        """Test outlier detection using z-score method."""
        outliers = detect_outliers(self.sample_series, method='zscore', threshold=2)

        # Should detect extreme values
        self.assertTrue(outliers.iloc[3])  # 100 should be detected as outlier

    def test_validate_dataframe(self):
        """Test DataFrame validation function."""
        report = validate_dataframe(self.sample_df, required_columns=['A', 'B'])

        self.assertTrue(report['is_valid'])
        self.assertEqual(report['shape'], (5, 2))
        self.assertEqual(len(report['errors']), 0)

        # Test with missing required column
        report_missing = validate_dataframe(self.sample_df, required_columns=['A', 'B', 'C'])
        self.assertFalse(report_missing['is_valid'])
        self.assertTrue(len(report_missing['errors']) > 0)

class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_data_preprocessing_pipeline(self):
        """Test complete data preprocessing pipeline."""
        # This would be a more comprehensive test
        # For now, just test that the pipeline can be initialized
        data_loader = DataLoader()

        # Test configuration loading
        self.assertIsNotNone(data_loader.config)
        self.assertIn('preprocessing', data_loader.config)

if __name__ == '__main__':
    # Create test data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Run tests
    unittest.main(verbosity=2)