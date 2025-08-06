"""
Training pipeline for flood prediction LSTM models.

This module provides functions for model training, validation, and data preparation
including specialized preprocessing for hydrological time series data.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from typing import List, Tuple, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import FloodDataset


def train_epoch(model: torch.nn.Module, train_loader: DataLoader,
                optimizer: torch.optim.Optimizer, loss_fn: Callable) -> float:
    """Execute one training epoch with gradient updates.

    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for gradient updates
        loss_fn: Loss function for computing training loss

    Returns:
        Average training loss for the epoch
    """
    model.train()  # Enable training mode (dropout, batch norm, etc.)
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Clear gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def validate_epoch(model: torch.nn.Module, val_loader: DataLoader, loss_fn: Callable) -> float:
    """Execute one validation epoch without gradient updates.

    Args:
        model: Neural network model to validate
        val_loader: DataLoader for validation data
        loss_fn: Loss function for computing validation loss

    Returns:
        Average validation loss for the epoch
    """
    model.eval()  # Enable evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, targets in val_loader:
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                optimizer: torch.optim.Optimizer, loss_fn: Callable, num_epochs: int) -> Tuple[
    List[float], List[float]]:
    """Complete model training loop with validation tracking.

    Trains the model for specified number of epochs, tracking both training
    and validation losses for monitoring convergence.

    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for gradient updates
        loss_fn: Loss function for training
        num_epochs: Number of training epochs

    Returns:
        Tuple of (training_losses, validation_losses) lists
    """
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        train_losses.append(train_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss))

        # Validate after each epoch
        val_loss = validate_epoch(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        print(f'Validation Loss: {val_loss:.4f}')

    return train_losses, val_losses


def create_data_loaders(data: torch.Tensor, config) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders with temporal splitting.

    Splits time series data chronologically to prevent data leakage,
    ensuring validation data comes from later time periods than training data.

    Args:
        data: Combined feature and target tensor
        config: Configuration object with data parameters

    Returns:
        Tuple of (train_loader, validation_loader)
    """
    # Temporal split: earlier data for training, later data for validation
    split_idx = int(len(data) * config.train_portion)
    train_data = data[:split_idx].float()
    valid_data = data[split_idx:].float()

    # Create dataset objects with sequence windowing
    train_dataset = FloodDataset(
        train_data,
        sequence_length=config.sequence_length,
        forecast_length=config.forecast_length
    )
    valid_dataset = FloodDataset(
        valid_data,
        sequence_length=config.sequence_length,
        forecast_length=config.forecast_length
    )

    # Create data loaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, valid_loader


def prepare_training_data(gauge_df, label_col: str = 'flow_rate', config=None):
    """Prepare and clean hydrological data for LSTM training.

    Performs comprehensive data preprocessing including missing value handling,
    feature engineering, and quality filtering for hydrological time series.

    Args:
        gauge_df: Raw gauge and rainfall data DataFrame
        label_col: Name of target variable column
        config: Configuration object (optional)

    Returns:
        Tuple of (processed_data_tensor, target_series)
    """
    # Replace string missing value indicators with NaN
    gauge_df = gauge_df.replace('-', np.nan)

    # Remove records with missing flow rate measurements
    gauge_df = gauge_df[~gauge_df['flow_rate'].isna()]

    # Ensure all rainfall stations have valid measurements
    rainfall_columns = ['ims_1_rain', 'ims_2_rain', 'ims_3_rain', 'ims_4_rain', 'ims_5_rain']
    for col in rainfall_columns:
        gauge_df = gauge_df[~gauge_df[col].isna()]

    # Restrict to time period with complete data coverage
    valid_start = gauge_df.apply(pd.Series.first_valid_index).max()
    valid_end = gauge_df.apply(pd.Series.last_valid_index).min()
    gauge_df = gauge_df.loc[valid_start:valid_end].reset_index(drop=True)

    # Remove metadata columns not needed for modeling
    metadata_cols = ['datetime', 'gauge_id', 'ims_1_id', 'ims_2_id', 'ims_3_id', 'ims_4_id', 'ims_5_id']
    gauge_df = gauge_df.drop(columns=metadata_cols)

    # Convert distance measurements from kilometers to meters
    distance_columns = ['ims_1_dist', 'ims_2_dist', 'ims_3_dist', 'ims_4_dist', 'ims_5_dist']
    gauge_df[distance_columns] *= 1000

    # Separate features and targets
    train_features = gauge_df.drop(columns=[label_col])
    train_targets = gauge_df[label_col]

    # Combine features and target into single array for dataset creation
    combined_data = np.hstack((
        train_features.values,
        train_targets.values.reshape(-1, 1)
    )).astype(np.float32)

    train_data = torch.tensor(combined_data).float()

    return train_data, train_targets
