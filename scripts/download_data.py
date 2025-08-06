#!/usr/bin/env python3
"""
Data collection and preprocessing pipeline for flood prediction system.

This script handles the complete data pipeline from IMS API data collection
through gauge-rainfall integration and final dataset preparation.
"""

import os
import sys
import pandas as pd

# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import load_config
from data_pipeline import IMSDataCollector, load_gauge_data, create_station_intersection_mapping, \
    create_integrated_dataset


def download_ims_data(config):
    """Download meteorological station data from IMS API.

    Args:
        config: System configuration object
    """
    print("Initializing IMS data collector...")
    collector = IMSDataCollector(config.api_token)

    print("Downloading rainfall data from all active stations...")
    collector.batch_download_all_stations()

    print("Consolidating individual station files...")
    # Combine all individual station files into single dataset
    station_files = [
        f"rain_station_data/{file_name}"
        for file_name in os.listdir("rain_station_data")
        if file_name.endswith('.csv')
    ]

    if not station_files:
        print("Warning: No station data files found. Check API connectivity.")
        return False

    station_dfs = [pd.read_csv(file_path) for file_path in station_files]
    consolidated_df = pd.concat(station_dfs, ignore_index=True)

    os.makedirs("data_saved", exist_ok=True)
    consolidated_df.to_csv("data_saved/IMS_data_final.csv", index=False)

    print(f"IMS data collection completed: {len(consolidated_df)} total records")
    return True


def process_gauge_data(config):
    """Process gauge flow data and create integrated datasets.

    Args:
        config: System configuration object
    """
    print("Processing river gauge and rainfall integration...")

    # Load consolidated rainfall data
    try:
        station_dfs = pd.read_csv("data_saved/IMS_data_final.csv")
    except FileNotFoundError:
        print("Error: IMS data not found. Run IMS data download first.")
        return False

    # Load gauge flow measurements
    try:
        gauge_df = load_gauge_data("Hydrographs/Hydrograph_201011_201819.csv")
    except FileNotFoundError:
        print("Error: Gauge hydrograph file not found.")
        print("Please ensure 'Hydrographs/Hydrograph_201011_201819.csv' exists.")
        return False

    # Load spatial intersection mapping
    try:
        gauge_ims_map = pd.read_csv("gauge_IMS_intersection/intersection.csv")
    except FileNotFoundError:
        print("Error: Gauge-IMS intersection mapping not found.")
        print("Please ensure 'gauge_IMS_intersection/intersection.csv' exists.")
        return False

    # Process each target gauge station
    print(f"Processing {len(config.gauge_ids)} gauge stations...")
    data_quality_summary = {}

    for gauge_id in config.gauge_ids:
        print(f"Processing gauge station {gauge_id}...")

        result_df = create_integrated_dataset(
            gauge_id, gauge_df, station_dfs, gauge_ims_map
        )

        if result_df is not None:
            complete_records = result_df.dropna().shape[0]
            total_records = len(result_df)
            completeness = (complete_records / total_records) * 100 if total_records > 0 else 0

            data_quality_summary[gauge_id] = {
                'total_records': total_records,
                'complete_records': complete_records,
                'completeness_percent': completeness
            }

            print(f"  Processed {total_records} total records, {complete_records} complete ({completeness:.1f}%)")
        else:
            print(f"  Failed to process gauge {gauge_id}")

    # Save data quality summary
    quality_df = pd.DataFrame([
        {'gauge_id': gid, **stats}
        for gid, stats in data_quality_summary.items()
    ])
    quality_df.to_csv('dfs_per_gauge_full_data/data_quality_summary.csv', index=False)

    print(f"Gauge data processing completed for {len(data_quality_summary)} stations")
    return True


def create_intersection_mapping():
    """Create spatial intersection mapping from GIS analysis if needed."""
    gis_file = "gauge_IMS_intersection/gis_project_generate_near_station_output.csv"
    output_file = "gauge_IMS_intersection/intersection.csv"

    if os.path.exists(output_file):
        print("Spatial intersection mapping already exists")
        return True

    if not os.path.exists(gis_file):
        print(f"Warning: GIS analysis file not found: {gis_file}")
        print("Spatial mapping will need to be created manually")
        return False

    print("Creating spatial intersection mapping from GIS analysis...")
    try:
        create_station_intersection_mapping(gis_file, output_file)
        print("Spatial intersection mapping created successfully")
        return True
    except Exception as e:
        print(f"Error creating intersection mapping: {e}")
        return False


def main():
    """Execute complete data pipeline."""
    print("Flash Flood Prediction System - Data Pipeline")
    print("=" * 50)

    # Load configuration
    config = load_config()
    print(f"Target gauge stations: {config.gauge_ids}")

    # Ensure required directories exist
    directories = [
        "rain_station_data",
        "data_saved",
        "dfs_per_gauge_full_data",
        "gauge_IMS_intersection"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    success_count = 0
    total_steps = 3

    try:
        # Step 1: Create spatial intersection mapping
        print(f"\nStep 1/{total_steps}: Spatial Intersection Mapping")
        if create_intersection_mapping():
            success_count += 1

        # Step 2: Download IMS meteorological data
        print(f"\nStep 2/{total_steps}: IMS Data Collection")
        if download_ims_data(config):
            success_count += 1

        # Step 3: Process and integrate gauge data
        print(f"\nStep 3/{total_steps}: Data Integration")
        if process_gauge_data(config):
            success_count += 1

        # Final status
        print("\n" + "=" * 50)
        if success_count == total_steps:
            print("Data pipeline completed successfully")
            print("Next steps:")
            print("  - Run 'python scripts/train_model.py' to train the LSTM model")
            print("  - Check 'dfs_per_gauge_full_data/' for processed datasets")
        else:
            print(f"Pipeline partially completed ({success_count}/{total_steps} steps)")
            print("Please check error messages above and retry failed steps")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        raise


if __name__ == "__main__":
    main()