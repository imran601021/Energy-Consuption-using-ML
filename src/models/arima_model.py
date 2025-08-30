import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import warnings
import yaml
import logging
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAModel:
    """ARIMA model for energy consumption prediction."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ARIMA model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['models']['arima']
        self.model = None
        self.fitted_model = None
        self.order = None
        self.seasonal_order = None

    def check_stationarity(self, ts: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
        """Check if time series is stationary using Augmented Dickey-Fuller test."""
        logger.info("Checking stationarity...")

        # Perform ADF test
        adf_result = adfuller(ts.dropna())

        results = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < significance_level
        }

        logger.info(f"ADF Statistic: {results['adf_statistic']:.6f}")
        logger.info(f"p-value: {results['p_value']:.6f}")
        logger.info(f"Is stationary: {results['is_stationary']}")

        return results

    def make_stationary(self, ts: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """Make time series stationary by differencing."""
        logger.info("Making time series stationary...")

        original_ts = ts.copy()
        diff_count = 0

        while diff_count <= max_diff:
            stationarity_result = self.check_stationarity(ts)

            if stationarity_result['is_stationary']:
                logger.info(f"Time series is stationary after {diff_count} differences")
                return ts, diff_count

            if diff_count < max_diff:
                ts = ts.diff().dropna()
                diff_count += 1
            else:
                break

        logger.warning(f"Time series may not be stationary after {max_diff} differences")
        return ts, diff_count

    def seasonal_decomposition(self, ts: pd.Series, period: int = 24, save_plot: bool = True) -> Dict[str, pd.Series]:
        """Perform seasonal decomposition of time series."""
        logger.info("Performing seasonal decomposition...")

        # Ensure we have enough data points
        if len(ts) < 2 * period:
            logger.warning(f"Not enough data for seasonal decomposition with period {period}")
            return {}

        decomposition = seasonal_decompose(
            ts.dropna(), 
            model='additive', 
            period=period
        )

        if save_plot:
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))

            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')

            plt.tight_layout()
            plt.savefig('results/plots/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
            plt.show()

        return {
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }

    def plot_acf_pacf(self, ts: pd.Series, lags: int = 40, save_plot: bool = True) -> None:
        """Plot ACF and PACF for order determination."""
        logger.info("Plotting ACF and PACF...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        plot_acf(ts.dropna(), lags=lags, ax=ax1, title='Autocorrelation Function')
        plot_pacf(ts.dropna(), lags=lags, ax=ax2, title='Partial Autocorrelation Function')

        plt.tight_layout()

        if save_plot:
            plt.savefig('results/plots/acf_pacf.png', dpi=300, bbox_inches='tight')

        plt.show()

    def auto_arima_order_selection(self, ts: pd.Series, seasonal: bool = True, 
                                  period: int = 24) -> Dict[str, Any]:
        """Automatically determine optimal ARIMA order using auto_arima."""
        logger.info("Performing automatic ARIMA order selection...")

        try:
            auto_model = auto_arima(
                ts.dropna(),
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=seasonal,
                m=period if seasonal else 1,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )

            order = auto_model.order
            seasonal_order = auto_model.seasonal_order if seasonal else None

            logger.info(f"Optimal ARIMA order: {order}")
            if seasonal_order:
                logger.info(f"Optimal seasonal order: {seasonal_order}")

            return {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': auto_model.aic(),
                'bic': auto_model.bic(),
                'model': auto_model
            }

        except Exception as e:
            logger.error(f"Auto ARIMA failed: {str(e)}")
            # Fallback to configured order
            return {
                'order': tuple(self.model_config['order']),
                'seasonal_order': tuple(self.model_config['seasonal_order']) if 'seasonal_order' in self.model_config else None
            }

    def train(self, ts: pd.Series, auto_order: bool = True, 
              seasonal: bool = False, period: int = 24) -> None:
        """Train ARIMA model."""
        logger.info("Training ARIMA model...")

        # Check and make stationary if needed
        stationarity_result = self.check_stationarity(ts)

        # Determine model order
        if auto_order:
            order_result = self.auto_arima_order_selection(ts, seasonal, period)
            self.order = order_result['order']
            self.seasonal_order = order_result.get('seasonal_order')
        else:
            self.order = tuple(self.model_config['order'])
            self.seasonal_order = tuple(self.model_config['seasonal_order']) if seasonal else None

        # Fit ARIMA model
        try:
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model = SARIMAX(
                    ts.dropna(),
                    order=self.order,
                    seasonal_order=self.seasonal_order
                )
            else:
                self.model = ARIMA(ts.dropna(), order=self.order)

            self.fitted_model = self.model.fit()

            logger.info(f"ARIMA model fitted successfully with order: {self.order}")
            if self.seasonal_order:
                logger.info(f"Seasonal order: {self.seasonal_order}")

            # Print model summary
            logger.info(f"AIC: {self.fitted_model.aic}")
            logger.info(f"BIC: {self.fitted_model.bic}")

        except Exception as e:
            logger.error(f"ARIMA model fitting failed: {str(e)}")
            raise

    def predict(self, n_periods: int, return_conf_int: bool = False, 
                alpha: float = 0.05) -> Dict[str, Any]:
        """Make predictions using fitted ARIMA model."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet!")

        logger.info(f"Making predictions for {n_periods} periods...")

        try:
            # Make forecast
            forecast_result = self.fitted_model.forecast(
                steps=n_periods,
                alpha=alpha
            )

            if return_conf_int:
                conf_int = self.fitted_model.get_forecast(n_periods).conf_int(alpha=alpha)
                return {
                    'forecast': forecast_result,
                    'conf_int_lower': conf_int.iloc[:, 0],
                    'conf_int_upper': conf_int.iloc[:, 1]
                }
            else:
                return {'forecast': forecast_result}

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_in_sample(self) -> pd.Series:
        """Get in-sample predictions (fitted values)."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet!")

        return self.fitted_model.fittedvalues

    def plot_forecast(self, ts: pd.Series, n_periods: int = 48, 
                     save_plot: bool = True) -> None:
        """Plot forecast with confidence intervals."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet!")

        # Make forecast with confidence intervals
        forecast_result = self.predict(n_periods, return_conf_int=True)

        plt.figure(figsize=(15, 8))

        # Plot original data (last 200 points)
        ts_plot = ts.iloc[-200:] if len(ts) > 200 else ts
        plt.plot(ts_plot.index, ts_plot.values, label='Observed', color='blue')

        # Create forecast index
        last_date = ts.index[-1]
        if isinstance(last_date, pd.Timestamp):
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=n_periods,
                freq='H'
            )
        else:
            forecast_index = range(len(ts), len(ts) + n_periods)

        # Plot forecast
        plt.plot(forecast_index, forecast_result['forecast'], 
                label='Forecast', color='red', linestyle='--')

        # Plot confidence intervals
        plt.fill_between(
            forecast_index,
            forecast_result['conf_int_lower'],
            forecast_result['conf_int_upper'],
            color='red', alpha=0.2, label='Confidence Interval'
        )

        plt.title('ARIMA Forecast')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_plot:
            plt.savefig('results/plots/arima_forecast.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_residuals(self, save_plot: bool = True) -> None:
        """Plot model residuals for diagnostic checking."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet!")

        residuals = self.fitted_model.resid

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals plot
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].grid(True)

        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True)

        # ACF of residuals
        plot_acf(residuals, ax=axes[1, 1], title='ACF of Residuals')

        plt.tight_layout()

        if save_plot:
            plt.savefig('results/plots/arima_residuals.png', dpi=300, bbox_inches='tight')

        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save the fitted ARIMA model."""
        if self.fitted_model is None:
            raise ValueError("No model to save!")

        model_data = {
            'fitted_model': self.fitted_model,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'model_config': self.model_config
        }

        joblib.dump(model_data, filepath)
        logger.info(f"ARIMA model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a saved ARIMA model."""
        model_data = joblib.load(filepath)

        self.fitted_model = model_data['fitted_model']
        self.order = model_data['order']
        self.seasonal_order = model_data['seasonal_order']
        self.model_config = model_data['model_config']

        logger.info(f"ARIMA model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        if self.fitted_model is None:
            return {"status": "Model not fitted"}

        return {
            "model_type": "ARIMA" if self.seasonal_order is None else "SARIMA",
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "log_likelihood": self.fitted_model.llf,
            "n_observations": self.fitted_model.nobs
        }

class EnsembleARIMA:
    """Ensemble of ARIMA models with different orders."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.models = []
        self.weights = []

    def train_ensemble(self, ts: pd.Series, orders: List[Tuple] = None) -> None:
        """Train ensemble of ARIMA models."""
        if orders is None:
            orders = [
                (1, 1, 1),
                (2, 1, 1), 
                (1, 1, 2),
                (2, 1, 2),
                (3, 1, 1)
            ]

        logger.info(f"Training ensemble of {len(orders)} ARIMA models...")

        model_scores = []

        for order in orders:
            try:
                model = ARIMAModel(self.config_path)
                model.order = order
                model.seasonal_order = None

                # Fit model
                arima_model = ARIMA(ts.dropna(), order=order)
                fitted_model = arima_model.fit()

                model.fitted_model = fitted_model

                # Calculate AIC for weighting
                aic = fitted_model.aic
                model_scores.append(aic)
                self.models.append(model)

                logger.info(f"ARIMA{order} - AIC: {aic:.2f}")

            except Exception as e:
                logger.warning(f"Failed to fit ARIMA{order}: {str(e)}")
                model_scores.append(float('inf'))

        # Calculate weights (inverse of AIC, normalized)
        valid_scores = [score for score in model_scores if score != float('inf')]
        if valid_scores:
            min_aic = min(valid_scores)
            # Convert AIC to weights (lower AIC = higher weight)
            raw_weights = [1 / (score - min_aic + 1) if score != float('inf') else 0 
                          for score in model_scores]
            total_weight = sum(raw_weights)
            self.weights = [w / total_weight if total_weight > 0 else 0 for w in raw_weights]

        logger.info("Ensemble training completed!")

    def predict(self, n_periods: int) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("Ensemble not trained yet!")

        predictions = []
        for model, weight in zip(self.models, self.weights):
            if weight > 0:
                try:
                    pred = model.predict(n_periods)['forecast']
                    predictions.append(pred.values * weight)
                except:
                    continue

        if predictions:
            return np.sum(predictions, axis=0)
        else:
            raise ValueError("No valid predictions from ensemble models!")

if __name__ == "__main__":
    # Example usage
    from src.data_processing.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    data = loader.load_processed_data()

    # Train ARIMA model on target variable
    arima_model = ARIMAModel()
    arima_model.train(data['y_train'], auto_order=True)

    # Make predictions
    forecast = arima_model.predict(n_periods=24)

    print("ARIMA model training completed!")
    print(f"Model info: {arima_model.get_model_info()}")