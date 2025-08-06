"""
Utility functions for model management, visualization, and data quality analysis.

This module provides helper functions for model persistence, training visualization,
data validation, and performance monitoring.
"""

import os
import sys
# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict


def plot_training_history(train_losses: List[float], val_losses: List[float],
                         title: str = "Training and Validation Loss using NSE_tag") -> None:
    """Visualize training and validation loss curves over epochs.

    Creates a line plot showing the convergence behavior of both training
    and validation losses throughout the training process.

    Args:
        train_losses: List of training loss values per epoch
        val_losses: List of validation loss values per epoch
        title: Plot title for the loss curves
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save trained model state dictionary to file.

    Args:
        model: Trained PyTorch model
        path: File path for saving model weights
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class: type, path: str, **model_kwargs) -> torch.nn.Module:
    """Load pre-trained model from saved state dictionary.

    Args:
        model_class: Model class constructor
        path: Path to saved model weights
        **model_kwargs: Model initialization parameters

    Returns:
        Loaded model with pre-trained weights
    """
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {path}")
    return model


def calculate_data_quality_stats(df: pd.DataFrame) -> Dict[str, int]:
    """Analyze data quality metrics for a DataFrame.

    Computes comprehensive statistics about missing values, data completeness,
    and overall dataset quality for monitoring and validation.

    Args:
        df: Input DataFrame to analyze

    Returns:
        Dictionary containing data quality metrics
    """
    stats = {'total_rows': len(df), 'total_columns': len(df.columns), 'rows_with_missing': df.isna().any(axis=1).sum(),
             'complete_rows': df.dropna().shape[0],
             'missing_value_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
             'columns_with_missing': df.isna().any().sum(), 'missing_by_column': df.isna().sum().to_dict()}

    # Add per-column missing value counts

    return stats


def create_loss_function(s_b: float):
    """Create NSE* loss function with gauge-specific standard deviation.

    Factory function that creates a customized loss function incorporating
    the standard deviation of observed flow for a specific gauge station.

    Args:
        s_b: Standard deviation of observed flow rates from training data

    Returns:
        Configured NSE* loss function
    """
    from model import NSE_tag

    def nse_tag_loss(y_pred, y_true):
        """NSE* loss function with embedded standard deviation."""
        return NSE_tag(y_pred, y_true, s_b)

    return nse_tag_loss


def print_model_summary(model: torch.nn.Module, input_shape: tuple) -> None:
    """Display detailed model architecture and parameter information.

    Args:
        model: PyTorch model to summarize
        input_shape: Expected input tensor shape (sequence_length, features)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"Model Type: {model.__class__.__name__}")
    print(f"Input Shape: {input_shape}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}")
    print("=" * 60)
    print("Layer Details:")
    print(model)
    print("=" * 60)


def validate_data_integrity(df: pd.DataFrame, gauge_id: int) -> bool:
    """Validate processed dataset for required columns and data integrity.

    Performs comprehensive validation to ensure all required features are present
    and the dataset is suitable for model training.

    Args:
        df: Processed dataset DataFrame
        gauge_id: Gauge station identifier for error reporting

    Returns:
        True if validation passes, False otherwise
    """
    required_columns = [
        'datetime', 'gauge_id', 'flow_rate',
        'ims_1_rain', 'ims_2_rain', 'ims_3_rain', 'ims_4_rain', 'ims_5_rain',
        'ims_1_dist', 'ims_2_dist', 'ims_3_dist', 'ims_4_dist', 'ims_5_dist'
    ]

    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns for gauge {gauge_id}: {missing_cols}")
        return False

    # Check for empty dataset
    if df.empty:
        print(f"ERROR: Empty dataset for gauge {gauge_id}")
        return False

    # Check for reasonable data ranges
    if 'flow_rate' in df.columns:
        flow_stats = df['flow_rate'].describe()
        if flow_stats['min'] < 0:
            print(f"WARNING: Negative flow rates detected for gauge {gauge_id}")

        rainfall_cols = [col for col in df.columns if 'rain' in col]
        for col in rainfall_cols:
            if df[col].min() < 0:
                print(f"WARNING: Negative rainfall values in {col} for gauge {gauge_id}")

    print(f"Data validation passed for gauge {gauge_id}")
    return True
