import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
import yaml
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestModel:
    """Random Forest model for energy consumption prediction."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Random Forest model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['models']['random_forest']
        self.model = None
        self.feature_importance = None

    def create_model(self) -> RandomForestRegressor:
        """Create Random Forest model with configured parameters."""
        logger.info("Creating Random Forest model...")

        model = RandomForestRegressor(
            n_estimators=self.model_config['n_estimators'],
            max_depth=self.model_config['max_depth'],
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config['n_jobs']
        )

        return model

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearch with TimeSeriesSplit."""
        logger.info("Performing hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Create base model
        rf = RandomForestRegressor(
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config['n_jobs']
        )

        # Perform grid search
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {-grid_search.best_score_}")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              tune_hyperparameters: bool = False) -> None:
        """Train the Random Forest model."""
        logger.info("Training Random Forest model...")

        if tune_hyperparameters:
            # Perform hyperparameter tuning
            tuning_results = self.hyperparameter_tuning(X_train, y_train)
            self.model = tuning_results['best_model']
        else:
            # Use default parameters
            self.model = self.create_model()
            self.model.fit(X_train, y_train)

        # Calculate feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        logger.info("Model training completed!")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save!")

        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance,
            'config': self.model_config
        }, filepath)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a saved model from disk."""
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.feature_importance = loaded_data['feature_importance']
        self.model_config = loaded_data['config']

        logger.info(f"Model loaded from {filepath}")

    def plot_feature_importance(self, top_n: int = 20, save_path: str = None) -> None:
        """Plot feature importance."""
        if self.feature_importance is None:
            raise ValueError("Feature importance not available!")

        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)

        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        if self.model is None:
            return {"status": "Model not trained"}

        return {
            "model_type": "Random Forest Regressor",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "n_features": self.model.n_features_in_,
            "feature_importance_available": self.feature_importance is not None,
            "top_5_features": self.feature_importance.head().to_dict() if self.feature_importance is not None else None
        }

class EnsembleRandomForest:
    """Ensemble of Random Forest models for improved predictions."""

    def __init__(self, n_models: int = 5, config_path: str = "config/config.yaml"):
        """Initialize ensemble of Random Forest models."""
        self.n_models = n_models
        self.models = []
        self.config_path = config_path

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train ensemble of Random Forest models."""
        logger.info(f"Training ensemble of {self.n_models} Random Forest models...")

        for i in range(self.n_models):
            logger.info(f"Training model {i+1}/{self.n_models}")

            # Create model with different random state
            model = RandomForestModel(self.config_path)
            model.model_config['random_state'] = 42 + i

            # Bootstrap sampling
            sample_indices = np.random.choice(
                len(X_train), 
                size=int(len(X_train) * 0.8), 
                replace=True
            )

            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]

            model.train(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions by averaging individual model predictions."""
        if not self.models:
            raise ValueError("Ensemble not trained yet!")

        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

    def save_ensemble(self, filepath: str) -> None:
        """Save the ensemble models."""
        ensemble_data = {
            'models': self.models,
            'n_models': self.n_models,
            'config_path': self.config_path
        }
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")

    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble models."""
        ensemble_data = joblib.load(filepath)
        self.models = ensemble_data['models']
        self.n_models = ensemble_data['n_models']
        self.config_path = ensemble_data['config_path']
        logger.info(f"Ensemble loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    from src.data_processing.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    data = loader.load_processed_data()

    # Train Random Forest model
    rf_model = RandomForestModel()
    rf_model.train(
        data['X_train'], 
        data['y_train'], 
        data['X_val'], 
        data['y_val'],
        tune_hyperparameters=False
    )

    # Make predictions
    train_pred = rf_model.predict(data['X_train'])
    val_pred = rf_model.predict(data['X_val'])

    print("Random Forest model training completed!")
    print(f"Model info: {rf_model.get_model_info()}")