#!/usr/bin/env python3
"""
Main training script for household energy consumption prediction models.
This script trains multiple models and compares their performance.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing.data_loader import DataLoader
from src.models.random_forest_model import RandomForestModel, EnsembleRandomForest
from src.models.lstm_model import LSTMModel, MultivariateLSTM
from src.models.arima_model import ARIMAModel, EnsembleARIMA
from src.evaluation.model_evaluator import ModelEvaluator

# Setup logging
def setup_logging(config_path: str = "config/config.yaml"):
    """Setup logging configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create logs directory
        log_file = logging_config.get('file_path', 'logs/training.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to setup logging from config: {str(e)}")

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Complete pipeline for training and evaluating energy consumption models."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize training pipeline."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_loader = DataLoader(config_path)
        self.evaluator = ModelEvaluator(config_path)
        self.models = {}
        self.data = None

        # Create necessary directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for the project."""
        directories = [
            'data/raw',
            'data/processed',
            'results/models',
            'results/plots',
            'results/reports',
            'logs'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def load_and_prepare_data(self) -> Dict[str, Any]:
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")

        try:
            self.data = self.data_loader.load_processed_data()

            logger.info("Data loading completed successfully!")
            logger.info(f"Training data shape: {self.data['X_train'].shape}")
            logger.info(f"Validation data shape: {self.data['X_val'].shape}")
            logger.info(f"Test data shape: {self.data['X_test'].shape}")

            return self.data

        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def train_random_forest(self, tune_hyperparameters: bool = False) -> RandomForestModel:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")

        try:
            rf_model = RandomForestModel(self.config_path)
            rf_model.train(
                self.data['X_train'],
                self.data['y_train'],
                self.data['X_val'],
                self.data['y_val'],
                tune_hyperparameters=tune_hyperparameters
            )

            # Save model
            model_path = "results/models/random_forest_model.pkl"
            rf_model.save_model(model_path)

            # Plot feature importance
            rf_model.plot_feature_importance(
                save_path="results/plots/rf_feature_importance.png"
            )

            self.models['Random Forest'] = rf_model
            logger.info("Random Forest training completed!")

            return rf_model

        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            return None

    def train_ensemble_random_forest(self, n_models: int = 5) -> EnsembleRandomForest:
        """Train Ensemble Random Forest model."""
        logger.info(f"Training Ensemble Random Forest with {n_models} models...")

        try:
            ensemble_rf = EnsembleRandomForest(n_models, self.config_path)
            ensemble_rf.train(self.data['X_train'], self.data['y_train'])

            # Save ensemble
            ensemble_path = "results/models/ensemble_random_forest.pkl"
            ensemble_rf.save_ensemble(ensemble_path)

            self.models['Ensemble RF'] = ensemble_rf
            logger.info("Ensemble Random Forest training completed!")

            return ensemble_rf

        except Exception as e:
            logger.error(f"Ensemble Random Forest training failed: {str(e)}")
            return None

    def train_lstm(self, model_type: str = "univariate") -> LSTMModel:
        """Train LSTM model."""
        logger.info(f"Training LSTM model ({model_type})...")

        try:
            if model_type == "multivariate":
                lstm_model = MultivariateLSTM(self.config_path)
            else:
                lstm_model = LSTMModel(self.config_path)

            lstm_model.train(
                self.data['X_train'],
                self.data['y_train'],
                self.data['X_val'],
                self.data['y_val']
            )

            # Save model
            model_path = f"results/models/lstm_{model_type}_model.h5"
            lstm_model.save_model(model_path)

            # Plot training history
            lstm_model.plot_training_history(
                save_path=f"results/plots/lstm_{model_type}_training_history.png"
            )

            model_name = f"LSTM ({model_type.title()})"
            self.models[model_name] = lstm_model
            logger.info(f"LSTM {model_type} training completed!")

            return lstm_model

        except Exception as e:
            logger.error(f"LSTM {model_type} training failed: {str(e)}")
            return None

    def train_arima(self, auto_order: bool = True, seasonal: bool = False) -> ARIMAModel:
        """Train ARIMA model."""
        model_type = "SARIMA" if seasonal else "ARIMA"
        logger.info(f"Training {model_type} model...")

        try:
            arima_model = ARIMAModel(self.config_path)

            # ARIMA works with univariate time series (target variable only)
            arima_model.train(
                self.data['y_train'],
                auto_order=auto_order,
                seasonal=seasonal,
                period=24  # Assuming hourly data with daily seasonality
            )

            # Save model
            model_path = f"results/models/{model_type.lower()}_model.pkl"
            arima_model.save_model(model_path)

            # Plot forecast
            arima_model.plot_forecast(
                self.data['y_train'],
                n_periods=48,
                save_plot=True
            )

            # Plot residuals
            arima_model.plot_residuals(save_plot=True)

            self.models[model_type] = arima_model
            logger.info(f"{model_type} training completed!")

            return arima_model

        except Exception as e:
            logger.error(f"{model_type} training failed: {str(e)}")
            return None

    def train_ensemble_arima(self) -> EnsembleARIMA:
        """Train Ensemble ARIMA model."""
        logger.info("Training Ensemble ARIMA...")

        try:
            ensemble_arima = EnsembleARIMA(self.config_path)
            ensemble_arima.train_ensemble(self.data['y_train'])

            # Save ensemble
            ensemble_path = "results/models/ensemble_arima.pkl"
            joblib.dump(ensemble_arima, ensemble_path)

            self.models['Ensemble ARIMA'] = ensemble_arima
            logger.info("Ensemble ARIMA training completed!")

            return ensemble_arima

        except Exception as e:
            logger.error(f"Ensemble ARIMA training failed: {str(e)}")
            return None

    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate all trained models."""
        logger.info("Evaluating all trained models...")

        results = {}

        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")

                # Determine model type for evaluation
                if 'Random Forest' in model_name or 'RF' in model_name:
                    model_type = "sklearn"
                elif 'LSTM' in model_name:
                    model_type = "lstm"
                elif 'ARIMA' in model_name:
                    model_type = "arima"
                else:
                    # Try to determine based on model methods
                    if hasattr(model, 'predict') and hasattr(model, 'fit'):
                        model_type = "sklearn"
                    else:
                        model_type = "custom"

                # Evaluate model
                metrics = self.evaluator.evaluate_model(
                    model,
                    self.data['X_test'],
                    self.data['y_test'],
                    model_name,
                    model_type
                )

                results[model_name] = metrics

            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {str(e)}")
                continue

        # Generate comparison and plots
        if results:
            comparison_df = self.evaluator.compare_models(save_plot=True)

            # Generate additional plots
            self.evaluator.plot_predictions_vs_actual(
                save_path="results/plots/all_models_predictions.png"
            )
            self.evaluator.plot_residuals(
                save_path="results/plots/all_models_residuals.png"
            )

            # Generate evaluation report
            report = self.evaluator.generate_evaluation_report(
                save_path="results/reports/evaluation_report.txt"
            )

            logger.info("Model evaluation completed!")
            return comparison_df

        else:
            logger.error("No models were successfully evaluated!")
            return pd.DataFrame()

    def run_full_pipeline(self, 
                         train_rf: bool = True,
                         train_ensemble_rf: bool = False,
                         train_lstm: bool = True,
                         train_multivariate_lstm: bool = False,
                         train_arima: bool = True,
                         train_sarima: bool = False,
                         train_ensemble_arima: bool = False,
                         tune_rf_hyperparameters: bool = False) -> Dict[str, Any]:
        """Run the complete training and evaluation pipeline."""
        logger.info("Starting complete training pipeline...")

        try:
            # Load and prepare data
            self.load_and_prepare_data()

            # Train models based on configuration
            if train_rf:
                self.train_random_forest(tune_hyperparameters=tune_rf_hyperparameters)

            if train_ensemble_rf:
                self.train_ensemble_random_forest()

            if train_lstm:
                self.train_lstm("univariate")

            if train_multivariate_lstm:
                self.train_lstm("multivariate")

            if train_arima:
                self.train_arima(auto_order=True, seasonal=False)

            if train_sarima:
                self.train_arima(auto_order=True, seasonal=True)

            if train_ensemble_arima:
                self.train_ensemble_arima()

            # Evaluate all models
            comparison_results = self.evaluate_all_models()

            # Get best model
            best_model = self.evaluator.get_best_model(metric="RMSE")

            logger.info("="*50)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*50)
            logger.info(f"Best performing model: {best_model}")
            logger.info("Check 'results/' directory for:")
            logger.info("  - Trained models: results/models/")
            logger.info("  - Plots and visualizations: results/plots/")
            logger.info("  - Evaluation report: results/reports/")
            logger.info("="*50)

            return {
                'models': self.models,
                'comparison_results': comparison_results,
                'best_model': best_model,
                'evaluator': self.evaluator
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the training pipeline."""
    # Setup logging
    setup_logging()

    logger.info("Household Energy Consumption Prediction - Model Training Pipeline")
    logger.info("="*70)

    try:
        # Initialize pipeline
        pipeline = ModelTrainingPipeline()

        # Run complete pipeline
        results = pipeline.run_full_pipeline(
            train_rf=True,                    # Train Random Forest
            train_ensemble_rf=False,          # Train Ensemble Random Forest
            train_lstm=True,                  # Train LSTM
            train_multivariate_lstm=False,    # Train Multivariate LSTM
            train_arima=True,                 # Train ARIMA
            train_sarima=False,               # Train SARIMA (seasonal)
            train_ensemble_arima=False,       # Train Ensemble ARIMA
            tune_rf_hyperparameters=False     # Tune RF hyperparameters (slow)
        )

        logger.info("Training pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()