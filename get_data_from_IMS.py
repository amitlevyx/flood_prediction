import json
import pickle
import os
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime


def get_stations_lla(stations, save=False):
    """
    it is in the form stations['location'] = {'latitude':... , 'longitude':...}
    we will transform it to stations['latitude'] and stations['longitude']
    :param save:
    :param stations:
    :return:
    """
    stations['latitude'] = stations['location'].apply(lambda x: x['latitude'])
    stations['longitude'] = stations['location'].apply(lambda x: x['longitude'])
    stations.drop('location', axis=1, inplace=True)
    stations = stations[['stationId', 'latitude', 'longitude']].dropna()
    if save:
        stations.to_csv("data_saved/IMS_stations_lla.csv", index=False)
    return stations


def get_data_from_IMS():
    url = "https://api.ims.gov.il/v1/Envista/stations"
    headers = {"Authorization": "ApiToken f058958a-d8bd-47cc-95d7-7ecf98610e47"}

    response = requests.request("GET", url, headers=headers)
    station_data = pd.DataFrame(json.loads(response.text))
    station_data = station_data[station_data['active'] == True]
    stations_ids = get_stations_lla(station_data)['stationId'].unique()

    dfs = []
    for station in tqdm(stations_ids):
        if os.path.exists(f"rain_station_data/IMS_data_station_{station}.csv"):
            # print(f"already have data for station {station}")
            continue
        print(f"getting data for station {station}")
        cur_df = pd.DataFrame()

        time_series_url = f"https://api.ims.gov.il/v1/envista/stations/{station}/data/1/?from=2010/10/01&to=2019/03/18"
        headers = {"Authorization": "ApiToken f058958a-d8bd-47cc-95d7-7ecf98610e47"}
        response = requests.request("GET", time_series_url, headers=headers)
        if not response.text:
            print(f"no data for station {station}")
            continue
        try:
            station_time_series = pd.DataFrame(json.loads(response.text))
        except:
            print(f"no data for station {station}")
            continue
        # for each row extract the data located in the 'data' column, and create a new column with the data
        station_time_series['datetime'] = station_time_series['data'].apply(lambda x: x['datetime'])
        cur_df['datetime'] = station_time_series['data'].apply(lambda x: x['datetime'])
        if len(station_time_series['data'].apply(lambda x: x['channels']).iloc[0]) == 0:
            print(f"no data for station {station}")
            continue
        cur_df['Rains'] = station_time_series['data'].apply(lambda x: x['channels'][0]['value'])
        cur_df['stationId'] = station
        cur_df.to_csv(f"rain_station_data/IMS_data_station_{station}.csv", index=False)
        dfs.append(cur_df)
        print(f"finished getting data for station {station}")

        # pickle.dump(dfs, open("data_saved/IMS_data.pkl", "wb"))
    return dfs


def map_stations_df2gauge_stations(station_df):
    station_gauge_map = pd.read_csv("gauge_IMS_intersection/intersection.csv")
    df_merged = pd.merge(station_df, station_gauge_map, on='stationId', how='left')
    df_merged.rename(columns={'gauge_id': 'gauge'}, inplace=True)
    df_merged.dropna(subset=['gauge'], inplace=True)
    return df_merged[['stationId', 'gauge', 'datetime', 'Rains']]


def create_intersection_csv_from_GenerateNear(filename):
    station_gauge_map = pd.read_csv(filename)

    cols = ['gauge_id', 'ims_1', 'dist_1', 'ims_2', 'dist_2', 'ims_3', 'dist_3', 'ims_4', 'dist_4', 'ims_5', 'dist_5', ]
    new_data_map = pd.DataFrame(columns=cols)
    # for each gauge id, we have in the original data multiple rows with the ims stations that are close to it.
    # we want to create a row for each gauge id that states the 5 closest ims stations and the distances
    for gauge_id in station_gauge_map['il_basin_shape_G_ExportTable.gauge_id'].unique():
        cur_df = station_gauge_map[station_gauge_map['il_basin_shape_G_ExportTable.gauge_id'] == gauge_id]
        cur_df.sort_values(by=['il_basin_shape_G_ExportTable.NEAR_DIST'], inplace=True)
        cur_df = cur_df.head(5)

        data = {f"ims_{int(i + 1)}": cur_df['IMS_stations_lla_pint.stationId'].iloc[i] for i in
                range(cur_df.shape[0])}
        data.update({f"dist_{i + 1}": cur_df['il_basin_shape_G_ExportTable.NEAR_DIST'].iloc[i] for i in
                     range(cur_df.shape[0])})
        data['gauge_id'] = gauge_id
        new_data_map = new_data_map.append(pd.DataFrame(data, index=[0]), ignore_index=True)
    new_data_map.to_csv("gauge_IMS_intersection/intersection_1.csv", index=False)

def create_df_for_gauge_id(gauge_id, gauge_df, gauge_ims_map, not_na_dict):
    if f'il_{int(gauge_id)}' not in gauge_ims_map['gauge_id'].unique():
        print(f"no intersection for gauge {gauge_id}")
        return
    cur_gauge_df = gauge_df[gauge_df['gauge_id'] == gauge_id]
    cur_gauge_df = cur_gauge_df.sort_values(by='flow_rate', ascending=False).drop_duplicates(subset=['gauge_id', 'datetime'], keep='first')
    cur_gauge_df = cur_gauge_df.sort_values(by='datetime')
    full_cur_gauge_df = cur_gauge_df.set_index('datetime').resample('10T').asfreq().interpolate(method='time', limit_area='inside')
    full_cur_gauge_df = full_cur_gauge_df.reset_index()
    # convert the datetime column to object type so we can merge with the station_dfs. save it in the format 2019-10-03T00:40:00+03:00
    full_cur_gauge_df['datetime'] = full_cur_gauge_df['datetime'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S')).astype(
        str)
    for indx in range(1, 6):
        cur_ims_id = gauge_ims_map[gauge_ims_map['gauge_id'] == f'il_{gauge_id}'][f'ims_{indx}'].values[0]
        cur_ims_dist = gauge_ims_map[gauge_ims_map['gauge_id'] == f'il_{gauge_id}'][f'dist_{indx}'].values[0]
        cur_ims_df = station_dfs[station_dfs['stationId'] == cur_ims_id]
        cur_ims_df.rename(columns={'stationId': f'ims_{indx}_id', 'Rains': f'ims_{indx}_rain'}, inplace=True)
        cur_ims_df = cur_ims_df.replace(-9999, np.nan)
        # cur_ims_df['datetime'] = pd.to_datetime(cur_ims_df['datetime'])
        # cur_ims_df = cur_ims_df.set_index('datetime').resample('10T').asfreq().interpolate(method='time', limit_area='inside').reset_index()
        # cur_ims_df['datetime'] = cur_ims_df['datetime'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S')).astype(str)
        # cur_ims_df.loc['datetime'] = pd.to_datetime(cur_ims_df['datetime'])
        # cur_ims_df = cur_ims_df.set_index('datetime').resample('10T').mean().interpolate(method='linear')
        full_cur_gauge_df = full_cur_gauge_df.merge(cur_ims_df, on='datetime', how='left')
        full_cur_gauge_df[f'ims_{indx}_dist'] = cur_ims_dist
        if cur_ims_id is None:
            full_cur_gauge_df[f'ims_{indx}_id'] = False
            full_cur_gauge_df[f'ims_{indx}_rain'] = False
        if cur_ims_df.shape[0] > 0:
            full_cur_gauge_df[f'ims_{indx}_id'] = cur_ims_id

    # gauge_dfs.append(cur_gauge_df)
    not_na_dict[gauge_id] = full_cur_gauge_df.dropna().shape[0]
    full_cur_gauge_df.loc[(full_cur_gauge_df['ims_1_rain'] == 0) & (full_cur_gauge_df['ims_2_rain'] == 0) & (full_cur_gauge_df['ims_3_rain'] == 0) & (
                full_cur_gauge_df['ims_4_rain'] == 0) & (full_cur_gauge_df['ims_5_rain'] == 0), 'flow_rate'] = 0
    full_cur_gauge_df.to_csv('dfs_per_gauge_full_data/{}.csv'.format(gauge_id), index=False)

if __name__ == "__main__":
    # station_dfs = get_data_from_IMS()
    # exit()
    # station_dfs = [pd.read_csv(f"rain_station_data/{file_name}") for file_name in os.listdir("rain_station_data")]
    # station_dfs = pd.concat(station_dfs)
    # station_dfs.to_csv("data_saved/IMS_data_final.csv", index=False)
    # create_intersection_csv_from_GenerateNear("gauge_IMS_intersection/GIS_project/il_basin_shapes_GenerateNear2_TableToExcel.csv")
    # create_intersection_csv_from_GenerateNear("gauge_IMS_intersection/amit_final.csv")
    #
    station_dfs = pd.read_csv("data_saved/IMS_data_final.csv")
    gauge_df = pd.read_csv("Hydrographs/Hydrograph_201011_201819.csv", encoding = "ISO-8859-8", parse_dates=['זמן מדידת ספיקה'])
    gauge_df.rename(columns={'זיהוי תחנה': 'gauge_id', "זמן מדידת ספיקה": "datetime", "ספיקה (מ''ק/שנייה)": "flow_rate"}, inplace=True)
    gauge_df = gauge_df[['gauge_id', 'datetime', 'flow_rate']]
    # # use linear interpolation on gauge_df to fill missing values
    # gauge_df = gauge_df.groupby('gauge_id').apply(lambda x: x.set_index('datetime').resample('10T').mean().interpolate(method='linear'))
    # gauge_df.reset_index(inplace=True)
    gauge_ims_map = pd.read_csv("gauge_IMS_intersection/intersection_1.csv")
    not_na_dict = {}
    for gauge_id in [2105, 5110, 7105]:
        create_df_for_gauge_id(gauge_id, gauge_df, gauge_ims_map, not_na_dict)
    pd.DataFrame(not_na_dict.items(), columns=['gauge_id', 'not_na']).to_csv('dfs_per_gauge_full_data/not_na.csv', index=False)
    #

    #
    # df.to_csv("data_saved/IMS_data_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S")), index=False)
    #
    # print(gauge_df[['gauge_id', 'datetime', 'flow_rate']])
