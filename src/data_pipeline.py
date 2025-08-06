"""
Data pipeline for Flash Flood Prediction System.

This module handles data collection from the Israeli Meteorological Service API,
processes hydrograph data, and creates integrated datasets combining rainfall
measurements with river flow data for machine learning training.
"""

import json
import os
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class IMSDataCollector:
    """Client for Israeli Meteorological Service API data collection.

    Handles authentication, station discovery, and time series data download
    from the IMS Envista API system.
    """

    def __init__(self, api_token: str):
        """Initialize API client with authentication token.

        Args:
            api_token: Valid IMS API authentication token
        """
        self.api_token = api_token
        self.base_url = "https://api.ims.gov.il/v1"
        self.headers = {"Authorization": f"ApiToken {api_token}"}

    def get_active_stations(self) -> pd.DataFrame:
        """Retrieve all active meteorological stations with geographic coordinates.

        Returns:
            DataFrame with columns: stationId, latitude, longitude
        """
        url = f"{self.base_url}/Envista/stations"
        response = requests.request("GET", url, headers=self.headers)
        station_data = pd.DataFrame(json.loads(response.text))

        # Filter for active stations only
        station_data = station_data[station_data['active'] == True]

        # Extract geographic coordinates from nested location structure
        station_data['latitude'] = station_data['location'].apply(lambda x: x['latitude'])
        station_data['longitude'] = station_data['location'].apply(lambda x: x['longitude'])
        station_data.drop('location', axis=1, inplace=True)

        # Return clean coordinate data
        stations = station_data[['stationId', 'latitude', 'longitude']].dropna()
        return stations

    def download_station_data(self, station_id: int, start_date: str = "2010/10/01",
                              end_date: str = "2019/03/18") -> Optional[pd.DataFrame]:
        """Download rainfall time series data for a specific station.

        Args:
            station_id: IMS station identifier
            start_date: Data collection start date (YYYY/MM/DD format)
            end_date: Data collection end date (YYYY/MM/DD format)

        Returns:
            DataFrame with columns: datetime, Rains, stationId
            None if no data available
        """
        time_series_url = f"{self.base_url}/envista/stations/{station_id}/data/1/?from={start_date}&to={end_date}"
        response = requests.request("GET", time_series_url, headers=self.headers)

        if not response.text:
            print(f"No data available for station {station_id}")
            return None

        try:
            station_time_series = pd.DataFrame(json.loads(response.text))
        except json.JSONDecodeError:
            print(f"Invalid JSON response for station {station_id}")
            return None

        # Extract time series data from nested API response structure
        station_time_series['datetime'] = station_time_series['data'].apply(lambda x: x['datetime'])

        # Check if station has rainfall measurement channels
        if len(station_time_series['data'].apply(lambda x: x['channels']).iloc[0]) == 0:
            print(f"No measurement channels for station {station_id}")
            return None

        # Create clean time series DataFrame
        cur_df = pd.DataFrame()
        cur_df['datetime'] = station_time_series['data'].apply(lambda x: x['datetime'])
        cur_df['Rains'] = station_time_series['data'].apply(lambda x: x['channels'][0]['value'])
        cur_df['stationId'] = station_id

        return cur_df

    def batch_download_all_stations(self, output_dir: str = "rain_station_data") -> None:
        """Download data for all active stations with progress tracking.

        Args:
            output_dir: Directory to save individual station CSV files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        stations = self.get_active_stations()
        stations_ids = stations['stationId'].unique()

        for station in tqdm(stations_ids, desc="Downloading station data"):
            output_file = f"{output_dir}/IMS_data_station_{station}.csv"

            # Skip if already downloaded
            if os.path.exists(output_file):
                continue

            print(f"Downloading data for station {station}")
            station_df = self.download_station_data(station)

            if station_df is not None:
                station_df.to_csv(output_file, index=False)
                print(f"Saved data for station {station}")


def load_gauge_data(file_path: str) -> pd.DataFrame:
    """Load and standardize river gauge hydrograph data.

    Processes Hebrew-encoded CSV files from Israeli water authority,
    standardizing column names and data types.

    Args:
        file_path: Path to hydrograph CSV file

    Returns:
        DataFrame with columns: gauge_id, datetime, flow_rate
    """
    gauge_df = pd.read_csv(file_path, encoding="ISO-8859-8", parse_dates=['זמן מדידת ספיקה'])

    # Standardize Hebrew column names to English
    gauge_df.rename(columns={
        'זיהוי תחנה': 'gauge_id',
        "זמן מדידת ספיקה": "datetime",
        "ספיקה (מ''ק/שנייה)": "flow_rate"
    }, inplace=True)

    return gauge_df[['gauge_id', 'datetime', 'flow_rate']]


def create_station_intersection_mapping(gis_file: str, output_file: str) -> None:
    """Create gauge-to-IMS station spatial mapping from GIS analysis output.

    Processes GIS "Generate Near" analysis results to identify the 5 closest
    rainfall stations for each river gauge, along with distances.

    Args:
        gis_file: Path to GIS analysis output CSV
        output_file: Path for processed intersection mapping
    """
    station_gauge_map = pd.read_csv(gis_file)

    cols = ['gauge_id', 'ims_1', 'dist_1', 'ims_2', 'dist_2', 'ims_3', 'dist_3', 'ims_4', 'dist_4', 'ims_5', 'dist_5']
    new_data_map = pd.DataFrame(columns=cols)

    # Process each gauge station to find 5 nearest rainfall stations
    for gauge_id in station_gauge_map['il_basin_shape_G_ExportTable.gauge_id'].unique():
        cur_df = station_gauge_map[station_gauge_map['il_basin_shape_G_ExportTable.gauge_id'] == gauge_id]
        cur_df.sort_values(by=['il_basin_shape_G_ExportTable.NEAR_DIST'], inplace=True)
        cur_df = cur_df.head(5)  # Take 5 closest stations

        # Extract station IDs and distances
        data = {f"ims_{int(i + 1)}": cur_df['IMS_stations_lla_pint.stationId'].iloc[i] for i in range(cur_df.shape[0])}
        data.update(
            {f"dist_{i + 1}": cur_df['il_basin_shape_G_ExportTable.NEAR_DIST'].iloc[i] for i in range(cur_df.shape[0])})
        data['gauge_id'] = gauge_id

        new_data_map = new_data_map.append(pd.DataFrame(data, index=[0]), ignore_index=True)

    new_data_map.to_csv(output_file, index=False)


def create_integrated_dataset(gauge_id: int, gauge_df: pd.DataFrame,
                              station_dfs: pd.DataFrame, mapping_df: pd.DataFrame,
                              output_dir: str = 'dfs_per_gauge_full_data') -> Optional[pd.DataFrame]:
    """Create integrated dataset combining gauge flow data with rainfall measurements.

    Merges river flow measurements with rainfall data from 5 nearest stations,
    applying temporal resampling and hydrological business rules.

    Args:
        gauge_id: Target gauge station identifier
        gauge_df: River flow measurement data
        station_dfs: Combined rainfall station data
        mapping_df: Spatial mapping between gauges and rainfall stations, created using GIS analysis
        output_dir: Directory for output CSV files

    Returns:
        Integrated DataFrame with flow and rainfall features
        None if processing fails
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if spatial mapping exists for this gauge
    if f'il_{int(gauge_id)}' not in mapping_df['gauge_id'].unique():
        print(f"No spatial intersection mapping for gauge {gauge_id}")
        return None

    # Process gauge data: remove duplicates and sort chronologically
    cur_gauge_df = gauge_df[gauge_df['gauge_id'] == gauge_id]
    cur_gauge_df = cur_gauge_df.sort_values(by='flow_rate', ascending=False).drop_duplicates(
        subset=['gauge_id', 'datetime'], keep='first')
    cur_gauge_df = cur_gauge_df.sort_values(by='datetime')

    # Resample to 10-minute intervals with temporal interpolation
    full_cur_gauge_df = cur_gauge_df.set_index('datetime').resample('10T').asfreq().interpolate(
        method='time', limit_area='inside')
    full_cur_gauge_df = full_cur_gauge_df.reset_index()

    full_cur_gauge_df['datetime'] = full_cur_gauge_df['datetime'].apply(
        lambda x: x.strftime('%Y-%m-%dT%H:%M:%S')).astype(str)

    # Integrate rainfall data from 5 nearest stations
    for indx in range(1, 6):
        cur_ims_id = mapping_df[mapping_df['gauge_id'] == f'il_{gauge_id}'][f'ims_{indx}'].values[0]
        cur_ims_dist = mapping_df[mapping_df['gauge_id'] == f'il_{gauge_id}'][f'dist_{indx}'].values[0]

        # Extract and clean rainfall data for current station
        cur_ims_df = station_dfs[station_dfs['stationId'] == cur_ims_id].copy()
        cur_ims_df.rename(columns={'stationId': f'ims_{indx}_id', 'Rains': f'ims_{indx}_rain'}, inplace=True)
        cur_ims_df = cur_ims_df.replace(-9999, np.nan)  # Replace missing value codes

        # Merge rainfall data with gauge data
        full_cur_gauge_df = full_cur_gauge_df.merge(cur_ims_df, on='datetime', how='left')
        full_cur_gauge_df[f'ims_{indx}_dist'] = cur_ims_dist

        # Handle missing station data
        if cur_ims_id is None:
            full_cur_gauge_df[f'ims_{indx}_id'] = False
            full_cur_gauge_df[f'ims_{indx}_rain'] = False
        elif cur_ims_df.shape[0] > 0:
            full_cur_gauge_df[f'ims_{indx}_id'] = cur_ims_id

    # Apply hydrological business rule: zero flow when no rainfall detected
    zero_rain_mask = (
            (full_cur_gauge_df['ims_1_rain'] == 0) &
            (full_cur_gauge_df['ims_2_rain'] == 0) &
            (full_cur_gauge_df['ims_3_rain'] == 0) &
            (full_cur_gauge_df['ims_4_rain'] == 0) &
            (full_cur_gauge_df['ims_5_rain'] == 0))
    full_cur_gauge_df.loc[zero_rain_mask, 'flow_rate'] = 0

    # Save integrated dataset
    output_file = f'{output_dir}/{gauge_id}.csv'
    full_cur_gauge_df.to_csv(output_file, index=False)

    return full_cur_gauge_df