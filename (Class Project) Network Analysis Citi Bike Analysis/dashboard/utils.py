
import collections
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st


def cal_distance(longitude_start, longitude_end, latitude_start, latitude_end):

    R = 6373.0

    lat1 = np.radians(latitude_start)
    lon1 = np.radians(longitude_start)
    lat2 = np.radians(latitude_end)
    lon2 = np.radians(longitude_end)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c * 1000
    return distance


def node_positions(df):
    df1 = df[['start_station_id', 'start_lat', 'start_lng']].drop_duplicates()
    df1.columns = ['station_id', 'lat', 'lng']
    df2 = df[['end_station_id', 'end_lat', 'end_lng']].drop_duplicates()
    df2.columns = ['station_id', 'lat', 'lng']

    df3 = df1.append(df2, ignore_index=True).drop_duplicates()
    names_list = list(set(df3.station_id.to_list()))

    node_position_df = pd.DataFrame()
    for idx, content in enumerate(names_list):
        node_position_df = node_position_df.append(df3[df3['station_id'] == content].iloc[0])
    node_position_df = node_position_df.reset_index(drop=True)
    node_position_df.set_index('station_id', inplace=True)
    node_position_df["pos"] = list(zip(node_position_df["lng"].astype(float), node_position_df["lat"].astype(float)))

    return node_position_df


@st.cache(allow_output_mutation=True)
def bike_network(df, node_position_df=None):

    G = nx.from_pandas_edgelist(df, 'start_station_id', 'end_station_id', edge_attr=['diff_time', 'distance'])

    edge_traffic = dict(collections.Counter(zip(df['start_station_id'].to_list(), df['end_station_id'].to_list())))
    nx.set_edge_attributes(G, edge_traffic, "edge_traffic")

    edge_features_df = df[["start_station_id", "end_station_id", "distance", "diff_time"]]
    edge_features_df = edge_features_df.groupby(["start_station_id", "end_station_id"], as_index=False).mean()
    distance_dict = {(row["start_station_id"], row["end_station_id"]): row["distance"]
                     for _, row in edge_features_df.iterrows()}
    duration_dict = {(row["start_station_id"], row["end_station_id"]): row["diff_time"]
                     for _, row in edge_features_df.iterrows()}
    nx.set_edge_attributes(G, distance_dict, "distance")
    nx.set_edge_attributes(G, duration_dict, "duration")

    # Positions
    stantions = node_positions(df) if node_position_df is None else node_position_df
    pos = stantions.to_dict()['pos']

    node_traffic = dict(collections.Counter((df['start_station_id'].to_list() + df['end_station_id'].to_list())))

    # Map "pos" atribute to nodes from pos dict
    mean_lng, std_lng = -74.04599297457915, 0.015399007166101202
    mean_lat, std_lat = 40.73090132373397, 0.013028667667883737
    for node, position in pos.items():
        try:
            G.nodes[node]['name'] = node
            G.nodes[node]['pos'] = position
            G.nodes[node]['lng'], G.nodes[node]['lat'] = G.nodes[node]['pos']
            G.nodes[node]['pos'] = ((G.nodes[node]['lng'] - mean_lng) / std_lng,
                                    (G.nodes[node]['lat'] - mean_lat) / std_lat)
            G.nodes[node]['lng_scaled'], G.nodes[node]['lat_scaled'] = G.nodes[node]['pos']
            G.nodes[node]['traffic'] = node_traffic[node]
            G.nodes[node]['area'] = node[:2]
        except KeyError:
            pass

    return G


def format_number(n):
    if n > 1e+6:
        return str(int(n // 1e+6)) + "m"
    elif n > 1e+3:
        return str(int(n // 1e+3)) + "k"
    return str(n)
