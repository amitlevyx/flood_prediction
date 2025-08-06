"""
Neural network models and evaluation metrics for flood prediction.

This module contains the LSTM architecture for time series forecasting
and specialized hydrological evaluation metrics including Nash-Sutcliffe
Efficiency (NSE), Kling-Gupta Efficiency (KGE), and custom variants.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.stats import pearsonr
import numpy as np


class FloodLSTM(nn.Module):
    """LSTM neural network for flood prediction time series modeling.

    Multi-layer LSTM architecture designed for sequence-to-sequence prediction
    of river flow rates from meteorological input features.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """Initialize LSTM flood prediction model.

        Args:
            input_size: Number of input features (rainfall + distance data)
            hidden_size: LSTM hidden layer dimension
            num_layers: Number of stacked LSTM layers
            output_size: Prediction horizon length (number of time steps)
        """
        super(FloodLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer LSTM with batch-first processing
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Final linear layer for flow rate prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through LSTM network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predicted flow rates of shape (batch_size, output_size)
        """
        batch_size = x.size(0)

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).float()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).float()

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x.float(), (h0, c0))

        # Use final time step output for prediction
        predictions = self.fc(lstm_out[:, -1, :])

        return predictions


class FloodDataset(Dataset):
    """PyTorch Dataset for flood prediction time series data.

    Creates sliding window sequences from time series data for LSTM training,
    with configurable input sequence length and prediction horizon.
    """

    def __init__(self, data, sequence_length=144, forecast_length=24):
        """Initialize flood prediction dataset.

        Args:
            data: Combined feature and target data array
            sequence_length: Length of input sequences (in time steps)
            forecast_length: Length of prediction horizon (in time steps)
        """
        # Separate features (all columns except last) from targets (last column)
        self.data = data[:, :-1]
        self.labels = data[:, -1]
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length

    def __len__(self):
        """Calculate number of valid sequence windows in dataset."""
        return len(self.data) - self.sequence_length - self.forecast_length + 1

    def __getitem__(self, idx):
        """Get input sequence and corresponding target values.

        Args:
            idx: Sample index

        Returns:
            tuple: (input_sequence, target_sequence)
        """
        # Extract target sequence starting after input sequence
        target_start = idx + self.sequence_length
        target_end = target_start + self.forecast_length
        label = self.labels[target_start:target_end]

        # Extract input sequence
        sequence = self.data[idx:idx + self.sequence_length]

        return sequence, label


def NSE(y_pred, y_true):
    """Calculate Nash-Sutcliffe Efficiency coefficient.

    NSE measures the relative magnitude of residual variance compared to
    the variance of observations. Perfect prediction yields NSE = 1.

    Args:
        y_pred: Predicted values tensor
        y_true: Observed values tensor

    Returns:
        NSE coefficient (higher is better, max = 1)
    """
    mean_observed = torch.mean(y_true)
    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - mean_observed) ** 2)
    nse = 1 - (numerator / denominator)
    return nse


def persistNSE(y_pred, y_true, s_b):
    """Calculate persistence-based Nash-Sutcliffe Efficiency.

    Compares model predictions against a naive persistence model
    (assuming next value equals current value).

    Args:
        y_pred: Model predictions tensor
        y_true: Observed values tensor
        s_b: Baseline standard deviation (unused in this implementation)

    Returns:
        Persistence NSE coefficient
    """
    # Create persistence prediction (shift observed values by one step)
    persistence_pred = torch.cat((y_true[0].unsqueeze(0), y_true[:-1]), dim=0)

    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - persistence_pred) ** 2)
    persist_nse = 1 - (numerator / denominator)

    return persist_nse


def KGE(y_pred, y_true):
    """Calculate Kling-Gupta Efficiency coefficient.

    KGE decomposes NSE into correlation, bias, and variability components,
    providing more balanced evaluation of model performance.

    Args:
        y_pred: Predicted values tensor
        y_true: Observed values tensor

    Returns:
        KGE coefficient (higher is better, max = 1)
    """

    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # Convert to numpy for correlation calculation
    pred_np = y_pred_flat.detach().numpy()
    true_np = y_true_flat.detach().numpy()

    # Calculate Pearson correlation coefficient
    r = pearsonr(pred_np, true_np)[0]
    r = torch.tensor(r)

    # Calculate bias ratio (alpha) and variability ratio (beta)
    alpha = torch.std(y_pred_flat) / torch.std(y_true_flat)  # Variability ratio
    beta = torch.mean(y_pred_flat) / torch.mean(y_true_flat)  # Bias ratio

    # Compute KGE from correlation, variability, and bias components
    kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge


def NSE_tag(y_pred, y_true, s_b, eps=1e-5):
    """Calculate normalized Nash-Sutcliffe Efficiency loss function (NSE*).

    Custom loss function that normalizes NSE by the standard deviation of
    observed flow, making it suitable for gradient-based optimization.
    This formulation helps with training stability across different flow regimes.

    Args:
        y_pred: Predicted flow values tensor
        y_true: Observed flow values tensor
        s_b: Standard deviation of observed flow from training data
        eps: Small constant to prevent division by zero

    Returns:
        NSE* loss value (lower is better)
    """
    numerator = torch.sum((y_pred - y_true) ** 2)
    denominator = (s_b - eps) ** 2
    return numerator / denominator
