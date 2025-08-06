# src/__init__.py
"""
Flash Flood Prediction System

Advanced Machine Learning for Real-Time Hydrological Forecasting
Developed using state-of-the-art LSTM neural networks with multi-source 
meteorological data integration.
"""

__version__ = "1.0.0"
__author__ = "Amit Levy"
__email__ = "amitlevyx@gmail.com"

from .config import Config, load_config
from .model import FloodLSTM, NSE, KGE, NSE_tag, persistNSE
from .data_pipeline import IMSDataCollector
from .training import train_model, train_epoch, validate_epoch

__all__ = [
    'Config',
    'load_config', 
    'FloodLSTM',
    'NSE',
    'KGE', 
    'NSE_tag',
    'persistNSE',
    'IMSDataCollector',
    'train_model',
    'train_epoch',
    'validate_epoch'
]