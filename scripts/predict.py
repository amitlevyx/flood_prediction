#!/usr/bin/env python3
"""
Flood prediction and model evaluation script.

This script demonstrates model inference capabilities, performance evaluation,
and visualization of prediction results for trained flood prediction models.
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flood_config import load_config
from model import FloodLSTM, NSE, KGE
from training import prepare_training_data, create_data_loaders
from utils import load_model, calculate_data_quality_stats


def evaluate_model_performance(model, test_loader, gauge_id):
    """Evaluate trained model using hydrological performance metrics.

    Args:
        model: Trained FloodLSTM model
        test_loader: DataLoader for evaluation data
        gauge_id: Gauge station identifier

    Returns:
        Tuple of (predictions, targets, performance_metrics)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    print(f"Evaluating model performance on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            outputs = model(data)
            all_predictions.extend(outputs.numpy())
            all_targets.extend(targets.numpy())

    # Convert to tensors for metric calculations
    predictions = torch.tensor(np.array(all_predictions))
    targets = torch.tensor(np.array(all_targets))

    # Calculate hydrological performance metrics
    try:
        nse_score = NSE(predictions, targets).item()
        kge_score = KGE(predictions, targets).item()
    except Exception as e:
        print(f"Warning: Could not calculate all metrics: {e}")
        nse_score = float('nan')
        kge_score = float('nan')

    # Create performance metrics dictionary
    performance_metrics = {
        'NSE': nse_score,
        'KGE': kge_score
    }

    print(f"Model Performance Summary for Gauge {gauge_id}:")
    for metric, value in performance_metrics.items():
        if not np.isnan(value):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: N/A")

    return predictions, targets, performance_metrics


def visualize_predictions(predictions, targets, gauge_id, n_samples=1000):
    """Create comprehensive visualization of model predictions.

    Args:
        predictions: Model prediction tensor
        targets: Ground truth tensor
        gauge_id: Gauge station identifier
        n_samples: Number of samples to visualize
    """
    # Flatten and limit samples for visualization
    pred_flat = predictions.numpy().flatten()[:n_samples]
    true_flat = targets.numpy().flatten()[:n_samples]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Flood Prediction Results - Gauge Station {gauge_id}', fontsize=16)

    # Time series comparison
    axes[0, 0].plot(true_flat, label='Observed Flow', alpha=0.8, linewidth=1)
    axes[0, 0].plot(pred_flat, label='Predicted Flow', alpha=0.8, linewidth=1)
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Flow Rate (m³/s)')
    axes[0, 0].set_title('Time Series Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter plot: Predicted vs Observed
    axes[0, 1].scatter(true_flat, pred_flat, alpha=0.5, s=1)
    min_val, max_val = min(true_flat.min(), pred_flat.min()), max(true_flat.max(), pred_flat.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Observed Flow Rate (m³/s)')
    axes[0, 1].set_ylabel('Predicted Flow Rate (m³/s)')
    axes[0, 1].set_title('Predicted vs Observed')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Residual plot
    residuals = pred_flat - true_flat
    axes[1, 0].scatter(true_flat, residuals, alpha=0.5, s=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Observed Flow Rate (m³/s)')
    axes[1, 0].set_ylabel('Residuals (m³/s)')
    axes[1, 0].set_title('Residual Analysis')
    axes[1, 0].grid(True, alpha=0.3)

    # Distribution comparison
    axes[1, 1].hist(true_flat, bins=50, alpha=0.7, label='Observed', density=True)
    axes[1, 1].hist(pred_flat, bins=50, alpha=0.7, label='Predicted', density=True)
    axes[1, 1].set_xlabel('Flow Rate (m³/s)')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].set_title('Flow Rate Distributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'predictions/predictions_{gauge_id}.png', dpi=300)


def analyze_prediction_errors(predictions, targets, gauge_id):
    """Analyze prediction errors and identify patterns.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        gauge_id: Gauge station identifier
    """
    pred_flat = predictions.numpy().flatten()
    true_flat = targets.numpy().flatten()
    residuals = pred_flat - true_flat

    print(f"Error Analysis for Gauge {gauge_id}:")
    print(f"  Mean Error (Bias): {np.mean(residuals):.4f} m³/s")
    print(f"  Mean Absolute Error: {np.mean(np.abs(residuals)):.4f} m³/s")
    print(f"  Root Mean Square Error: {np.sqrt(np.mean(residuals ** 2)):.4f} m³/s")
    print(f"  Standard Deviation: {np.std(residuals):.4f} m³/s")

    # Flow regime analysis
    low_flow_mask = true_flat < np.percentile(true_flat, 25)
    high_flow_mask = true_flat > np.percentile(true_flat, 75)

    print(f"Flow Regime Performance:")
    print(f"  Low Flow MAE: {np.mean(np.abs(residuals[low_flow_mask])):.4f} m³/s")
    print(f"  High Flow MAE: {np.mean(np.abs(residuals[high_flow_mask])):.4f} m³/s")


def run_prediction_analysis(gauge_id, model_path=None):
    """Execute complete prediction and analysis workflow.

    Args:
        gauge_id: Target gauge station identifier
        model_path: Optional path to pre-trained model

    Returns:
        Dictionary containing analysis results
    """
    config = load_config()

    print(f"Loading data for gauge station {gauge_id}...")

    # Load and prepare evaluation data
    try:
        gauge_df = pd.read_csv(f'../dfs_per_gauge_full_data/{gauge_id}.csv')
    except FileNotFoundError:
        print(f"Error: Data file for gauge {gauge_id} not found.")
        print("Please run 'python scripts/download_data.py' first.")
        return None

    # Prepare data for model evaluation
    train_data, train_targets = prepare_training_data(gauge_df, config=config)
    train_loader, val_loader = create_data_loaders(train_data, config)

    print(f"Dataset: {train_data.shape[0]} total samples")

    # Initialize model architecture
    model = FloodLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size
    )

    # Load pre-trained weights if available
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model = load_model(FloodLSTM, model_path,
                           input_size=config.input_size,
                           hidden_size=config.hidden_size,
                           num_layers=config.num_layers,
                           output_size=config.output_size)
    else:
        print("Warning: No pre-trained model found - using random initialization")
        print("Note: For meaningful predictions, train the model first using 'train_model.py'")

    # Evaluate model performance
    predictions, targets, metrics = evaluate_model_performance(model, val_loader, gauge_id)

    # Perform detailed error analysis
    analyze_prediction_errors(predictions, targets, gauge_id)

    # Create comprehensive visualizations
    print("Generating prediction visualizations...")
    visualize_predictions(predictions, targets, gauge_id)

    # Analyze data quality
    quality_stats = calculate_data_quality_stats(gauge_df)
    print(f"Data Quality Summary:")
    print(f"  Total Records: {quality_stats['total_rows']:,}")
    print(f"  Complete Records: {quality_stats['complete_rows']:,}")
    print(f"  Data Completeness: {(quality_stats['complete_rows'] / quality_stats['total_rows'] * 100):.1f}%")

    return {
        'gauge_id': gauge_id,
        'predictions': predictions,
        'targets': targets,
        'metrics': metrics,
        'data_quality': quality_stats
    }


def main():
    """Main prediction and evaluation workflow."""
    print("Flash Flood Prediction System - Model Evaluation")
    print("=" * 50)

    config = load_config()

    # Configure evaluation parameters
    gauge_id = 7105  # Primary evaluation gauge
    model_path = f"models/{gauge_id}_model.pth"  # Optional: path to saved model

    print(f"Target Gauge Station: {gauge_id}")
    print(f"Model Configuration: {config.hidden_size}H-{config.num_layers}L LSTM")
    print(f"Forecast Horizon: {config.hours_to_forecast} hours ({config.forecast_length} steps)")

    try:
        os.makedirs('predictions', exist_ok=True)

        # Execute prediction analysis
        results = run_prediction_analysis(gauge_id, model_path)

        if results is None:
            print("Analysis failed - check data availability")
            return

        # Display final summary
        print("Evaluation completed successfully")
        print(f"Generated {len(results['predictions'])} predictions")

        # Highlight key performance metrics
        metrics = results['metrics']
        print("Key Performance Indicators:")
        if not np.isnan(metrics.get('NSE', np.nan)):
            nse_rating = "Excellent" if metrics['NSE'] > 0.8 else "Good" if metrics['NSE'] > 0.6 else "Fair"
            print(f"  Nash-Sutcliffe Efficiency: {metrics['NSE']:.3f} ({nse_rating})")

        if not np.isnan(metrics.get('KGE', np.nan)):
            print(f"  Kling-Gupta Efficiency: {metrics['KGE']:.3f}")

        print("Next Steps:")
        print("  - Review prediction visualizations above")
        print(f"  - Experiment with different gauge stations: {config.gauge_ids}")
        print("  - Retrain model with updated data for improved performance")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()
