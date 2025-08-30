import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for energy consumption prediction."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.evaluation_config = self.config['evaluation']
        self.results = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        logger.info(f"Calculating metrics for {model_name}...")

        # Ensure arrays are the same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

        # Remove any infinite or NaN values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            logger.error(f"No valid predictions for {model_name}")
            return {}

        metrics = {}

        # Mean Squared Error
        mse = mean_squared_error(y_true, y_pred)
        metrics['MSE'] = mse

        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        metrics['RMSE'] = rmse

        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)
        metrics['MAE'] = mae

        # R-squared Score
        r2 = r2_score(y_true, y_pred)
        metrics['R2'] = r2

        # Mean Absolute Percentage Error
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
            metrics['MAPE'] = mape
        except:
            # Calculate MAPE manually if sklearn version doesn't support it
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['MAPE'] = mape

        # Mean Error (Bias)
        me = np.mean(y_pred - y_true)
        metrics['ME'] = me

        # Normalized RMSE (as percentage of mean)
        nrmse = (rmse / np.mean(y_true)) * 100
        metrics['NRMSE_%'] = nrmse

        # Maximum Error
        max_error = np.max(np.abs(y_true - y_pred))
        metrics['MAX_ERROR'] = max_error

        # Explained Variance Score
        from sklearn.metrics import explained_variance_score
        evs = explained_variance_score(y_true, y_pred)
        metrics['EVS'] = evs

        logger.info(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str, model_type: str = "sklearn") -> Dict[str, Any]:
        """Evaluate a single model and store results."""
        logger.info(f"Evaluating {model_name} model...")

        try:
            # Make predictions based on model type
            if model_type == "sklearn":
                y_pred = model.predict(X_test)
            elif model_type == "lstm":
                y_pred = model.predict(X_test)
                # Handle sequence data for LSTM
                if len(y_pred) != len(y_test):
                    min_len = min(len(y_pred), len(y_test))
                    y_pred = y_pred[-min_len:]
                    y_test = y_test.iloc[-min_len:]
            elif model_type == "arima":
                forecast_result = model.predict(len(y_test))
                y_pred = forecast_result['forecast'].values if isinstance(forecast_result, dict) else forecast_result
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Calculate metrics
            metrics = self.calculate_metrics(y_test.values, y_pred, model_name)

            # Store results
            self.results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'actual': y_test.values,
                'model_type': model_type
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return {}

    def compare_models(self, save_plot: bool = True) -> pd.DataFrame:
        """Compare all evaluated models."""
        if not self.results:
            logger.error("No models evaluated yet!")
            return pd.DataFrame()

        logger.info("Comparing model performance...")

        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            metrics['Model'] = model_name
            comparison_data.append(metrics)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('Model', inplace=True)

        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('RMSE')

        logger.info("Model Comparison Results:")
        print(comparison_df)

        if save_plot:
            self.plot_model_comparison(comparison_df)

        return comparison_df

    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            save_path: str = "results/plots/model_comparison.png") -> None:
        """Plot model comparison metrics."""
        # Select key metrics for plotting
        key_metrics = ['RMSE', 'MAE', 'R2', 'MAPE']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(key_metrics):
            if metric in comparison_df.columns:
                ax = axes[i]

                if metric == 'R2':
                    # For R2, higher is better
                    colors = plt.cm.RdYlGn(comparison_df[metric])
                else:
                    # For RMSE, MAE, MAPE, lower is better
                    colors = plt.cm.RdYlGn_r(comparison_df[metric] / comparison_df[metric].max())

                bars = ax.bar(comparison_df.index, comparison_df[metric], color=colors)
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_predictions_vs_actual(self, model_names: List[str] = None, 
                                 n_points: int = 200,
                                 save_path: str = "results/plots/predictions_vs_actual.png") -> None:
        """Plot predictions vs actual values for selected models."""
        if not self.results:
            logger.error("No models evaluated yet!")
            return

        if model_names is None:
            model_names = list(self.results.keys())

        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 5 * n_models))

        if n_models == 1:
            axes = [axes]

        for i, model_name in enumerate(model_names):
            if model_name not in self.results:
                continue

            result = self.results[model_name]
            actual = result['actual'][-n_points:]
            predicted = result['predictions'][-n_points:]

            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[-min_len:]
            predicted = predicted[-min_len:]

            ax = axes[i]
            x = range(len(actual))

            ax.plot(x, actual, label='Actual', color='blue', alpha=0.7)
            ax.plot(x, predicted, label='Predicted', color='red', alpha=0.7, linestyle='--')

            ax.set_title(f'{model_name} - Predictions vs Actual (Last {min_len} points)')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Energy Consumption')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_residuals(self, model_names: List[str] = None,
                      save_path: str = "results/plots/residuals_analysis.png") -> None:
        """Plot residuals analysis for selected models."""
        if not self.results:
            logger.error("No models evaluated yet!")
            return

        if model_names is None:
            model_names = list(self.results.keys())

        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5 * n_models))

        if n_models == 1:
            axes = axes.reshape(1, -1)

        for i, model_name in enumerate(model_names):
            if model_name not in self.results:
                continue

            result = self.results[model_name]
            actual = result['actual']
            predicted = result['predictions']

            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[-min_len:]
            predicted = predicted[-min_len:]

            residuals = actual - predicted

            # Residuals vs Fitted
            axes[i, 0].scatter(predicted, residuals, alpha=0.6)
            axes[i, 0].axhline(y=0, color='red', linestyle='--')
            axes[i, 0].set_xlabel('Fitted Values')
            axes[i, 0].set_ylabel('Residuals')
            axes[i, 0].set_title(f'{model_name} - Residuals vs Fitted')
            axes[i, 0].grid(True, alpha=0.3)

            # Residuals Distribution
            axes[i, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[i, 1].set_xlabel('Residuals')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].set_title(f'{model_name} - Residuals Distribution')
            axes[i, 1].grid(True, alpha=0.3)

            # Add mean and std to histogram
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            axes[i, 1].axvline(mean_res, color='red', linestyle='--', 
                              label=f'Mean: {mean_res:.3f}')
            axes[i, 1].axvline(mean_res + std_res, color='orange', linestyle=':', 
                              label=f'+1 STD: {std_res:.3f}')
            axes[i, 1].axvline(mean_res - std_res, color='orange', linestyle=':', 
                              label=f'-1 STD')
            axes[i, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_evaluation_report(self, save_path: str = "results/reports/evaluation_report.txt") -> str:
        """Generate comprehensive evaluation report."""
        if not self.results:
            logger.error("No models evaluated yet!")
            return ""

        logger.info("Generating evaluation report...")

        report = []
        report.append("="*60)
        report.append("HOUSEHOLD ENERGY CONSUMPTION PREDICTION - MODEL EVALUATION REPORT")
        report.append("="*60)
        report.append("")

        # Model comparison
        comparison_df = self.compare_models(save_plot=False)

        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 30)
        report.append(comparison_df.to_string())
        report.append("")

        # Best model identification
        best_model_rmse = comparison_df.index[0]  # Already sorted by RMSE
        best_model_r2 = comparison_df.loc[comparison_df['R2'].idxmax()].name
        best_model_mae = comparison_df.loc[comparison_df['MAE'].idxmin()].name

        report.append("BEST MODELS BY METRIC")
        report.append("-" * 20)
        report.append(f"Best RMSE: {best_model_rmse} ({comparison_df.loc[best_model_rmse, 'RMSE']:.4f})")
        report.append(f"Best R²: {best_model_r2} ({comparison_df.loc[best_model_r2, 'R2']:.4f})")
        report.append(f"Best MAE: {best_model_mae} ({comparison_df.loc[best_model_mae, 'MAE']:.4f})")
        report.append("")

        # Individual model analysis
        for model_name, result in self.results.items():
            report.append(f"DETAILED ANALYSIS - {model_name.upper()}")
            report.append("-" * 40)

            metrics = result['metrics']
            for metric, value in metrics.items():
                report.append(f"{metric}: {value:.6f}")

            # Calculate additional insights
            actual = result['actual']
            predicted = result['predictions']

            min_len = min(len(actual), len(predicted))
            actual = actual[-min_len:]
            predicted = predicted[-min_len:]

            residuals = actual - predicted

            report.append(f"Residual Statistics:")
            report.append(f"  Mean Residual: {np.mean(residuals):.6f}")
            report.append(f"  Std Residual: {np.std(residuals):.6f}")
            report.append(f"  Min Residual: {np.min(residuals):.6f}")
            report.append(f"  Max Residual: {np.max(residuals):.6f}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)

        if comparison_df.loc[best_model_rmse, 'R2'] > 0.8:
            report.append(f"✓ {best_model_rmse} shows excellent predictive performance (R² > 0.8)")
        elif comparison_df.loc[best_model_rmse, 'R2'] > 0.6:
            report.append(f"• {best_model_rmse} shows good predictive performance (R² > 0.6)")
        else:
            report.append(f"⚠ {best_model_rmse} shows moderate predictive performance (R² ≤ 0.6)")
            report.append("  Consider feature engineering or alternative models")

        if comparison_df.loc[best_model_rmse, 'NRMSE_%'] < 10:
            report.append("✓ Normalized RMSE is acceptable (< 10%)")
        else:
            report.append("⚠ High normalized RMSE (≥ 10%) - consider model improvements")

        report.append("")
        report.append("="*60)

        report_text = "\n".join(report)

        # Save report
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Evaluation report saved to {save_path}")

        return report_text

    def get_best_model(self, metric: str = "RMSE") -> str:
        """Get the name of the best performing model based on specified metric."""
        if not self.results:
            logger.error("No models evaluated yet!")
            return None

        if metric.upper() == "R2":
            # For R2, higher is better
            best_score = -float('inf')
            best_model = None

            for model_name, result in self.results.items():
                score = result['metrics'].get(metric.upper(), -float('inf'))
                if score > best_score:
                    best_score = score
                    best_model = model_name
        else:
            # For other metrics, lower is better
            best_score = float('inf')
            best_model = None

            for model_name, result in self.results.items():
                score = result['metrics'].get(metric.upper(), float('inf'))
                if score < best_score:
                    best_score = score
                    best_model = model_name

        return best_model

    def clear_results(self) -> None:
        """Clear all evaluation results."""
        self.results = {}
        logger.info("Evaluation results cleared")

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()

    # Simulate some test data
    y_true = np.random.randn(100) + 10
    y_pred_model1 = y_true + np.random.randn(100) * 0.1  # Good model
    y_pred_model2 = y_true + np.random.randn(100) * 0.5  # Worse model

    # Calculate metrics
    metrics1 = evaluator.calculate_metrics(y_true, y_pred_model1, "Model 1")
    metrics2 = evaluator.calculate_metrics(y_true, y_pred_model2, "Model 2")

    print("Example evaluation completed!")