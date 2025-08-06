#!/usr/bin/env python3
"""
Training script for flood prediction LSTM model.

This script handles the complete training workflow including data loading,
model initialization, training execution, and results visualization.
"""

import os
import sys
import pandas as pd
import torch
import torch.optim as optim

# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flood_config import load_config
from model import FloodLSTM
from training import train_model, create_data_loaders, prepare_training_data
from utils import plot_training_history, save_model, create_loss_function, print_model_summary


def main():
    """Main training execution function."""
    # Load system configuration
    config = load_config()

    # Configure target gauge station (can be made configurable via CLI)
    gauge_id = 7105

    print(f"Starting flood prediction model training for gauge station {gauge_id}")
    print(
        f"Model configuration: Hidden Size={config.hidden_size}, Layers={config.num_layers}, Learning Rate={config.learning_rate}")

    # Load and prepare training data
    print("Loading and preprocessing data...")
    try:
        gauge_df = pd.read_csv(f'../dfs_per_gauge_full_data/{gauge_id}.csv')
        print(f"Loaded {len(gauge_df)} raw data records")
    except FileNotFoundError:
        print(f"Error: Data file for gauge {gauge_id} not found.")
        print("Please run 'python scripts/download_data.py' first to collect data.")
        return

    # Preprocess data for training
    train_data, train_targets = prepare_training_data(gauge_df, config=config)
    print(f"Preprocessed data shape: {train_data.shape}")

    # Create data loaders for batch processing
    train_loader, valid_loader = create_data_loaders(train_data, config)
    print(f"Created {len(train_loader)} training batches, {len(valid_loader)} validation batches")

    # Initialize LSTM model
    print("Initializing LSTM model...")
    model = FloodLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size
    )

    print_model_summary(model, (config.sequence_length, config.input_size))

    # Configure optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create custom NSE* loss function
    training_targets = train_targets[:int(len(train_targets) * config.train_portion)]
    s_b = training_targets.std().item()
    loss_fn = create_loss_function(s_b)

    print(f"Using NSE* loss function with s_b = {s_b:.4f}")

    # Execute training loop
    print(f"Starting training for {config.num_epochs} epochs...")

    train_losses, val_losses = train_model(
        model, train_loader, valid_loader, optimizer, loss_fn, config.num_epochs
    )

    print("Training completed successfully")

    # Display final results
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")

    # Check for overfitting
    if final_val_loss > final_train_loss * 1.5:
        print("Warning: Potential overfitting detected (validation loss >> training loss)")
    elif final_val_loss < final_train_loss:
        print("Good generalization: validation loss <= training loss")

    # Save model (optional - comment to disable)
    model_path = f'models/{gauge_id}_model.pth'
    os.makedirs('models', exist_ok=True)
    save_model(model, model_path)

    # Visualize training progress
    print("Displaying training history...")
    plot_training_history(train_losses, val_losses)

    print(f"Training completed for gauge station {gauge_id}")


if __name__ == "__main__":
    main()
