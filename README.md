# Flash Flood Prediction System - LSTM Neural Networks

**Deep Learning for Real-Time Hydrological Forecasting**

## Overview

A deep learning system for flash flood prediction using LSTM neural networks with 2-hour prediction horizons. The system integrates meteorological data from multiple sources and provides real-time forecasting capabilities with 10-minute temporal resolution.

**Tech Stack:** Python, PyTorch, Deep Learning, Time Series Analysis, REST APIs, GIS Analytics

## Key Features

- End-to-end ML pipeline from data collection to model inference
- Custom hydrological loss functions (NSE, KGE, NSE*) for time series prediction
- Real-time data integration from Israeli Meteorological Service API
- Multi-sensor spatial data fusion with distance-weighted rainfall aggregation
- Production-ready inference pipeline with automated error handling

## Technical Architecture

### Deep Learning Models
- Multi-layer LSTM networks with configurable hidden dimensions
- Sequence-to-sequence modeling (24-hour input → 2-hour forecast)
- Custom evaluation metrics for hydrological performance assessment
- Hyperparameter optimization through systematic grid search

### Data Processing Pipeline
- Automated data collection from meteorological APIs
- Time series preprocessing with missing data imputation
- Spatial analytics for optimal sensor network configuration
- Feature engineering incorporating rainfall patterns across 5-station networks

### Model Training & Evaluation
- Time-based train/validation splits to prevent data leakage
- Nash-Sutcliffe Efficiency and Kling-Gupta Efficiency metrics
- Comparison against persistence baselines
- Cross-validation framework for robust performance estimation

## Project Structure

```
flood_prediction/
├── scripts/
│   ├── train_model.py          # Main training script
│   ├── predict.py              # Model inference and evaluation
│   └── download_data.py        # Data collection from APIs
├── src/
│   ├── model.py                # LSTM architecture and evaluation metrics
│   ├── training.py             # Training loops and data preparation
│   ├── config.py               # System configuration parameters
│   └── utils.py                # Visualization and helper functions
├── data/
│   └── dfs_per_gauge_full_data/    # Time series data per gauge station
├── models/                     # Trained model checkpoints
└── README.md
```

## Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scipy matplotlib
```

### Training a Model
```bash
# Train LSTM model for gauge station 7105
python scripts/train_model.py

# Download fresh data from meteorological service
python scripts/download_data.py
```

### Running Predictions
```bash
# Evaluate trained model performance
python scripts/predict.py
```

## Model Configuration

Key parameters in `src/config.py`:

- `sequence_length = 144`: Input window (24 hours at 10-min intervals)
- `forecast_length = 12`: Prediction horizon (2 hours)
- `hidden_size = 64`: LSTM hidden layer dimension
- `num_layers = 2`: Number of stacked LSTM layers
- `learning_rate = 0.001`: Adam optimizer learning rate

## Data Sources

- **River Gauges:** Flow rate measurements at 10-minute intervals
- **Rainfall Sensors:** Precipitation data from 5 meteorological stations per gauge
- **Spatial Data:** Station coordinates and distance calculations
- **API Integration:** Real-time data from Israeli Meteorological Service

## Evaluation Metrics

- **Nash-Sutcliffe Efficiency (NSE):** Standard hydrological model performance
- **Kling-Gupta Efficiency (KGE):** Balanced correlation, bias, and variability assessment
- **NSE* (Custom):** Normalized loss function for gradient-based optimization
- **Persistence Comparison:** Baseline against naive forecasting methods

## Results

The system achieves strong performance on validation datasets with:
- Effective 2-hour advance warning for flood events
- Robust generalization across different temporal periods
- Real-time inference capabilities with sub-second latency
- Systematic improvement over persistence baselines

## Technical Notes

### LSTM Architecture
- Batch-first processing for efficient GPU utilization
- Zero-initialized hidden states for consistent training
- Linear output layer for direct flow rate prediction
- Configurable depth and width for model capacity tuning

### Training Pipeline
- Temporal data splitting to prevent lookahead bias
- Custom loss functions optimized for hydrological metrics
- Automated validation tracking and convergence monitoring
- Model checkpointing and training history visualization

### Data Quality
- Automated missing value detection and filtering
- Temporal alignment across heterogeneous data sources
- Statistical outlier detection and quality control
- Distance-weighted spatial interpolation for rainfall data

## Development Notes

- All models use temporal train/test splits (80/20) to maintain realistic evaluation
- Hyperparameter tuning conducted across 125+ configuration combinations
- Production deployment includes comprehensive error handling and logging
- Modular architecture supports easy extension to additional gauge stations
