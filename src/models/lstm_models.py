import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import yaml
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for energy consumption prediction."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize LSTM model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['models']['lstm']
        self.model = None
        self.history = None
        self.scaler = MinMaxScaler()

    def create_sequences(self, data: np.ndarray, target: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(target[i + sequence_length])

        return np.array(X), np.array(y)

    def prepare_lstm_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame = None, y_val: pd.Series = None,
                         X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """Prepare data for LSTM training."""
        logger.info("Preparing data for LSTM...")

        sequence_length = self.model_config['sequence_length']

        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_scaled = y_train.values.reshape(-1, 1)

        # Create sequences for training data
        X_train_seq, y_train_seq = self.create_sequences(
            X_train_scaled, y_train_scaled.flatten(), sequence_length
        )

        data_dict = {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'sequence_length': sequence_length,
            'n_features': X_train.shape[1]
        }

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_scaled = y_val.values.reshape(-1, 1)

            X_val_seq, y_val_seq = self.create_sequences(
                X_val_scaled, y_val_scaled.flatten(), sequence_length
            )

            data_dict.update({
                'X_val': X_val_seq,
                'y_val': y_val_seq
            })

        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            y_test_scaled = y_test.values.reshape(-1, 1)

            X_test_seq, y_test_seq = self.create_sequences(
                X_test_scaled, y_test_scaled.flatten(), sequence_length
            )

            data_dict.update({
                'X_test': X_test_seq,
                'y_test': y_test_seq
            })

        logger.info(f"LSTM training data shape: {X_train_seq.shape}")
        return data_dict

    def create_model(self, n_features: int) -> Sequential:
        """Create LSTM model architecture."""
        logger.info("Creating LSTM model architecture...")

        model = Sequential([
            LSTM(
                self.model_config['hidden_units'],
                return_sequences=True,
                input_shape=(self.model_config['sequence_length'], n_features)
            ),
            Dropout(self.model_config['dropout_rate']),

            LSTM(self.model_config['hidden_units'], return_sequences=True),
            Dropout(self.model_config['dropout_rate']),

            LSTM(self.model_config['hidden_units']),
            Dropout(self.model_config['dropout_rate']),

            Dense(25, activation='relu'),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=self.model_config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              save_best_model: bool = True) -> None:
        """Train the LSTM model."""
        logger.info("Training LSTM model...")

        # Prepare data for LSTM
        lstm_data = self.prepare_lstm_data(X_train, y_train, X_val, y_val)

        # Create model
        self.model = self.create_model(lstm_data['n_features'])

        # Print model summary
        self.model.summary()

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]

        if save_best_model:
            model_checkpoint = ModelCheckpoint(
                'results/models/best_lstm_model.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                save_weights_only=False
            )
            callbacks.append(model_checkpoint)

        # Train the model
        validation_data = (lstm_data['X_val'], lstm_data['y_val']) if X_val is not None else None

        self.history = self.model.fit(
            lstm_data['X_train'], lstm_data['y_train'],
            batch_size=self.model_config['batch_size'],
            epochs=self.model_config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("LSTM model training completed!")

    def predict(self, X: pd.DataFrame, return_sequences: bool = False) -> np.ndarray:
        """Make predictions using the trained LSTM model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Prepare data for prediction
        X_scaled = self.scaler.transform(X)
        sequence_length = self.model_config['sequence_length']

        if len(X_scaled) < sequence_length:
            raise ValueError(f"Input data must have at least {sequence_length} samples")

        # Create sequences
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)), sequence_length)

        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)

        return predictions.flatten()

    def predict_future(self, X_last: pd.DataFrame, n_steps: int = 24) -> np.ndarray:
        """Predict future values using the last known data."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        predictions = []
        sequence_length = self.model_config['sequence_length']

        # Use the last sequence_length samples as starting point
        current_sequence = self.scaler.transform(X_last.iloc[-sequence_length:])

        for _ in range(n_steps):
            # Reshape for prediction
            current_input = current_sequence.reshape(1, sequence_length, -1)

            # Predict next value
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)

            # Update sequence (assuming the first feature is the target)
            # This is a simplified approach - in practice, you'd need to predict all features
            next_features = np.zeros(current_sequence.shape[1])
            next_features[0] = next_pred  # Assuming first feature is the target

            # Shift the sequence
            current_sequence = np.vstack([current_sequence[1:], next_features])

        return np.array(predictions)

    def plot_training_history(self, save_path: str = None) -> None:
        """Plot training history."""
        if self.history is None:
            raise ValueError("No training history available!")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save!")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.model.save(filepath)

        # Save scaler separately
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"LSTM model saved to {filepath}")
        logger.info(f"Scaler saved to {scaler_path}")

    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        self.model = tf.keras.models.load_model(filepath)

        # Load scaler
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        logger.info(f"LSTM model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        if self.model is None:
            return {"status": "Model not trained"}

        return {
            "model_type": "LSTM",
            "sequence_length": self.model_config['sequence_length'],
            "hidden_units": self.model_config['hidden_units'],
            "dropout_rate": self.model_config['dropout_rate'],
            "total_parameters": self.model.count_params(),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape
        }

class MultivariateLSTM(LSTMModel):
    """Multivariate LSTM model for energy consumption prediction."""

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path)

    def create_multivariate_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences where target is the first column."""
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length), :])  # All features
            y.append(data[i + sequence_length, 0])  # First column as target

        return np.array(X), np.array(y)

    def prepare_multivariate_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame = None, y_val: pd.Series = None,
                                X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """Prepare multivariate data for LSTM."""
        logger.info("Preparing multivariate data for LSTM...")

        sequence_length = self.model_config['sequence_length']

        # Combine features and target
        train_data = np.column_stack([y_train.values, X_train.values])
        train_data_scaled = self.scaler.fit_transform(train_data)

        # Create sequences
        X_train_seq, y_train_seq = self.create_multivariate_sequences(
            train_data_scaled, sequence_length
        )

        data_dict = {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'sequence_length': sequence_length,
            'n_features': train_data.shape[1]
        }

        if X_val is not None and y_val is not None:
            val_data = np.column_stack([y_val.values, X_val.values])
            val_data_scaled = self.scaler.transform(val_data)

            X_val_seq, y_val_seq = self.create_multivariate_sequences(
                val_data_scaled, sequence_length
            )

            data_dict.update({
                'X_val': X_val_seq,
                'y_val': y_val_seq
            })

        if X_test is not None and y_test is not None:
            test_data = np.column_stack([y_test.values, X_test.values])
            test_data_scaled = self.scaler.transform(test_data)

            X_test_seq, y_test_seq = self.create_multivariate_sequences(
                test_data_scaled, sequence_length
            )

            data_dict.update({
                'X_test': X_test_seq,
                'y_test': y_test_seq
            })

        return data_dict

if __name__ == "__main__":
    # Example usage
    from src.data_processing.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    data = loader.load_processed_data()

    # Train LSTM model
    lstm_model = LSTMModel()
    lstm_model.train(
        data['X_train'], 
        data['y_train'], 
        data['X_val'], 
        data['y_val']
    )

    # Make predictions
    predictions = lstm_model.predict(data['X_test'])

    print("LSTM model training completed!")
    print(f"Model info: {lstm_model.get_model_info()}")