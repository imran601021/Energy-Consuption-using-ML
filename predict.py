#!/usr/bin/env python3
"""
Prediction script for household energy consumption.
Load trained models and make predictions on new data.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing.data_loader import DataLoader
from src.models.random_forest_model import RandomForestModel
from src.models.lstm_model import LSTMModel
from src.models.arima_model import ARIMAModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyConsumptionPredictor:
    """Predictor for household energy consumption using trained models."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize predictor with configuration."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_loader = DataLoader(config_path)
        self.models = {}
        self.model_performance = {}

    def load_trained_models(self, model_dir: str = "results/models/") -> Dict[str, Any]:
        """Load all trained models from the models directory."""
        logger.info("Loading trained models...")

        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'LSTM': 'lstm_univariate_model.h5',
            'ARIMA': 'arima_model.pkl',
            'SARIMA': 'sarima_model.pkl'
        }

        loaded_models = {}

        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)

            if os.path.exists(model_path):
                try:
                    if model_name == 'Random Forest':
                        model = RandomForestModel(self.config_path)
                        model.load_model(model_path)
                        loaded_models[model_name] = model

                    elif model_name == 'LSTM':
                        model = LSTMModel(self.config_path)
                        model.load_model(model_path)
                        loaded_models[model_name] = model

                    elif model_name in ['ARIMA', 'SARIMA']:
                        model = ARIMAModel(self.config_path)
                        model.load_model(model_path)
                        loaded_models[model_name] = model

                    logger.info(f"✓ Loaded {model_name} model")

                except Exception as e:
                    logger.warning(f"✗ Failed to load {model_name}: {str(e)}")
            else:
                logger.warning(f"✗ Model file not found: {model_path}")

        self.models = loaded_models
        logger.info(f"Successfully loaded {len(loaded_models)} models")

        return loaded_models

    def prepare_prediction_data(self, data: pd.DataFrame, 
                              target_col: str = 'Global_active_power') -> Dict[str, Any]:
        """Prepare data for prediction using the same preprocessing as training."""
        logger.info("Preparing prediction data...")

        try:
            # Use the data loader's preprocessing pipeline
            # This ensures consistency with training data preparation
            processed_data = self.data_loader.preprocess_data()

            # If custom data is provided, we need to process it similarly
            if data is not None and not data.empty:
                # Apply same preprocessing steps
                # This is a simplified version - in practice, you'd want to ensure
                # the exact same preprocessing pipeline is applied
                logger.info("Processing custom data...")

                # Handle missing values
                data = self.data_loader.handle_missing_values(data)

                # Add time features if configured
                if self.data_loader.preprocessing_config['feature_engineering']['add_time_features']:
                    data = self.data_loader.add_time_features(data)

                # Add lag features if configured
                if self.data_loader.preprocessing_config['feature_engineering']['add_lag_features']:
                    data = self.data_loader.add_lag_features(data, target_col)

                # Add rolling features if configured
                if self.data_loader.preprocessing_config['feature_engineering']['rolling_window_features']:
                    data = self.data_loader.add_rolling_features(data, target_col)

                # Remove rows with NaN values
                data = data.dropna()

                # Separate features and target
                feature_cols = [col for col in data.columns if col != target_col]
                X = data[feature_cols]
                y = data[target_col] if target_col in data.columns else None

                return {
                    'X': X,
                    'y': y,
                    'feature_names': feature_cols,
                    'processed_data': data
                }

            else:
                # Use preprocessed test data
                return {
                    'X': processed_data['X_test'],
                    'y': processed_data['y_test'],
                    'feature_names': processed_data['feature_names'],
                    'processed_data': processed_data
                }

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

    def make_predictions(self, X: pd.DataFrame, 
                        models_to_use: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Make predictions using specified models."""
        if not self.models:
            raise ValueError("No models loaded! Call load_trained_models() first.")

        if models_to_use is None:
            models_to_use = list(self.models.keys())

        logger.info(f"Making predictions with {len(models_to_use)} models...")

        predictions = {}

        for model_name in models_to_use:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not available")
                continue

            try:
                model = self.models[model_name]

                if model_name == 'Random Forest':
                    pred = model.predict(X)
                elif model_name == 'LSTM':
                    pred = model.predict(X)
                elif model_name in ['ARIMA', 'SARIMA']:
                    # ARIMA models predict future values, not from features
                    # For this example, we'll predict the next len(X) values
                    forecast_result = model.predict(len(X))
                    pred = forecast_result['forecast'].values if isinstance(forecast_result, dict) else forecast_result

                predictions[model_name] = pred
                logger.info(f"✓ {model_name}: Generated {len(pred)} predictions")

            except Exception as e:
                logger.error(f"✗ Prediction failed for {model_name}: {str(e)}")

        return predictions

    def create_ensemble_prediction(self, predictions: Dict[str, np.ndarray], 
                                 weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Create ensemble prediction from individual model predictions."""
        if not predictions:
            raise ValueError("No predictions provided!")

        logger.info("Creating ensemble prediction...")

        # Default equal weights
        if weights is None:
            weights = {model: 1.0 / len(predictions) for model in predictions.keys()}

        # Ensure all predictions have the same length
        min_length = min(len(pred) for pred in predictions.values())

        ensemble_pred = np.zeros(min_length)
        total_weight = 0

        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            if weight > 0:
                ensemble_pred += pred[:min_length] * weight
                total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        logger.info(f"Ensemble prediction created using {len(predictions)} models")

        return ensemble_pred

    def predict_future(self, n_hours: int = 24, 
                      models_to_use: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predict future energy consumption for the next n_hours."""
        logger.info(f"Predicting energy consumption for next {n_hours} hours...")

        # Load the most recent data for context
        data = self.data_loader.load_processed_data()

        # Get last available data points
        X_last = data['X_test'].iloc[-100:]  # Use last 100 points as context

        future_predictions = {}

        for model_name, model in self.models.items():
            if models_to_use and model_name not in models_to_use:
                continue

            try:
                if model_name == 'LSTM':
                    # LSTM can predict future values
                    pred = model.predict_future(X_last, n_steps=n_hours)
                elif model_name in ['ARIMA', 'SARIMA']:
                    # ARIMA naturally predicts future values
                    forecast_result = model.predict(n_hours)
                    pred = forecast_result['forecast'].values if isinstance(forecast_result, dict) else forecast_result
                elif model_name == 'Random Forest':
                    # For Random Forest, we need to simulate future feature values
                    # This is a simplified approach - in practice, you'd need more sophisticated feature prediction
                    logger.warning(f"{model_name} cannot directly predict future without future features")
                    continue

                future_predictions[model_name] = pred
                logger.info(f"✓ {model_name}: Predicted {len(pred)} future values")

            except Exception as e:
                logger.error(f"✗ Future prediction failed for {model_name}: {str(e)}")

        # Create ensemble prediction
        if len(future_predictions) > 1:
            ensemble_pred = self.create_ensemble_prediction(future_predictions)
            future_predictions['Ensemble'] = ensemble_pred

        # Create time index for predictions
        last_timestamp = data['X_test'].index[-1]
        if isinstance(last_timestamp, pd.Timestamp):
            future_index = pd.date_range(
                start=last_timestamp + pd.Timedelta(hours=1),
                periods=n_hours,
                freq='H'
            )
        else:
            future_index = range(len(data['X_test']), len(data['X_test']) + n_hours)

        return {
            'predictions': future_predictions,
            'time_index': future_index,
            'n_hours': n_hours
        }

    def save_predictions(self, predictions: Dict[str, np.ndarray], 
                        time_index: pd.Index, 
                        filepath: str = "results/predictions.csv") -> None:
        """Save predictions to CSV file."""
        logger.info(f"Saving predictions to {filepath}...")

        # Create DataFrame
        pred_df = pd.DataFrame(predictions, index=time_index)

        # Save to CSV
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pred_df.to_csv(filepath)

        logger.info(f"Predictions saved successfully!")

    def plot_predictions(self, predictions: Dict[str, np.ndarray], 
                        time_index: pd.Index,
                        actual_values: Optional[np.ndarray] = None,
                        save_path: str = "results/plots/predictions.png") -> None:
        """Plot predictions."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 8))

        # Plot predictions
        for model_name, pred in predictions.items():
            plt.plot(time_index[:len(pred)], pred, label=f'{model_name} Prediction', 
                    marker='o', markersize=3, alpha=0.7)

        # Plot actual values if provided
        if actual_values is not None:
            min_len = min(len(actual_values), len(time_index))
            plt.plot(time_index[:min_len], actual_values[:min_len], 
                    label='Actual', color='black', linewidth=2, alpha=0.8)

        plt.title('Energy Consumption Predictions')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption (kW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Prediction plot saved to {save_path}")

def main():
    """Main function for prediction script."""
    logger.info("Household Energy Consumption Prediction Script")
    logger.info("="*50)

    try:
        # Initialize predictor
        predictor = EnergyConsumptionPredictor()

        # Load trained models
        models = predictor.load_trained_models()

        if not models:
            logger.error("No trained models found! Please run train_models.py first.")
            return

        # Example 1: Predict on test data
        logger.info("\nExample 1: Predicting on test data...")

        # Prepare test data
        data_info = predictor.prepare_prediction_data(None)  # Use default test data

        # Make predictions
        test_predictions = predictor.make_predictions(data_info['X'])

        # Plot predictions vs actual
        predictor.plot_predictions(
            test_predictions,
            data_info['X'].index,
            actual_values=data_info['y'].values,
            save_path="results/plots/test_predictions.png"
        )

        # Save predictions
        predictor.save_predictions(
            test_predictions,
            data_info['X'].index,
            filepath="results/test_predictions.csv"
        )

        # Example 2: Predict future values
        logger.info("\nExample 2: Predicting future 48 hours...")

        # Predict next 48 hours
        future_results = predictor.predict_future(
            n_hours=48,
            models_to_use=['LSTM', 'ARIMA']  # Only models that can predict future
        )

        # Plot future predictions
        predictor.plot_predictions(
            future_results['predictions'],
            future_results['time_index'],
            save_path="results/plots/future_predictions.png"
        )

        # Save future predictions
        predictor.save_predictions(
            future_results['predictions'],
            future_results['time_index'],
            filepath="results/future_predictions.csv"
        )

        logger.info("\nPrediction script completed successfully!")
        logger.info("Check 'results/' directory for:")
        logger.info("  - Prediction CSV files")
        logger.info("  - Prediction plots")

    except Exception as e:
        logger.error(f"Prediction script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()