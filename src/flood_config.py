"""
Configuration management for the Flash Flood Prediction System.

This module defines all model parameters, data processing settings, and system
configuration in a centralized, type-safe manner.
"""

from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import os


@dataclass
class Config:
    """Configuration class for flood prediction system parameters.

    Contains all hyperparameters, data processing settings, and system
    configuration needed for training and inference.
    """

    # Model architecture parameters
    hidden_size: int = 64  # LSTM hidden layer size
    num_layers: int = 2  # Number of LSTM layers
    learning_rate: float = 0.001  # Adam optimizer learning rate
    batch_size: int = 32  # Training batch size
    num_epochs: int = 1  # Number of training epochs

    # Temporal parameters for time series modeling
    hours_to_forecast: int = 2  # Prediction horizon in hours
    hours_to_base_on: int = 24  # Input sequence length in hours
    train_portion: float = 0.8  # Training/validation split ratio

    # Target gauge stations for modeling
    gauge_ids: List[int] = None

    # IMS API configuration. The token is assigned by the IMS per request, and is personal.
    load_dotenv()
    api_token: str = os.getenv("IMS_API_TOKEN")

    @property
    def forecast_length(self) -> int:
        """Calculate forecast length in 10-minute intervals."""
        return self.hours_to_forecast * 6

    @property
    def sequence_length(self) -> int:
        """Calculate input sequence length in 10-minute intervals."""
        return self.hours_to_base_on * 6

    @property
    def input_size(self) -> int:
        """Calculate input feature dimension.

        Features: 5 rainfall measurements + 5 distance values = 10 total
        """
        return 10

    @property
    def output_size(self) -> int:
        """Output dimension equals forecast length."""
        return self.forecast_length


def load_config() -> Config:
    """Load system configuration with default gauge stations.

    Returns:
        Config: Configured system parameters with default gauge stations
    """
    config = Config()
    config.gauge_ids = [2105, 5110, 7105]  # Primary gauge stations
    return config