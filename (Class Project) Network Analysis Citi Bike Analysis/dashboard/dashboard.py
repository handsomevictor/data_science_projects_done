
import os
import time
import numpy as np
import pandas as pd
import altair as alt
import networkx as nx
import nx_altair as nxa
import streamlit as st

from utils import cal_distance, node_positions, bike_network, format_number


st.write("# New York’s Citi Bikes Flow Analysis")


def load_month(year=2021, month_index=1):
    month_index = str(month_index)
    month_index = '0' + month_index if len(month_index) == 1 else month_index
    filename = f'./data/JC-{year}{month_index}-citibike-tripdata.csv'
    if os.path.isfile(filename):
        month_df = pd.read_csv(filename)
        if 'started_at' in month_df.columns:
            month_df = month_df[['started_at', 'ended_at',
                                 'start_station_name', 'start_station_id', 'end_station_name',
                                 'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng']]
        elif '###starttime' in month_df.columns:
            month_df = pd.DataFrame({
                'started_at': month_df["starttime"],
                'ended_at': month_df["stoptime"],
                'start_station_name': month_df["start station name"],
                'start_station_id': month_df["start station id"],
                'end_station_name': month_df["end station name"],
                'end_station_id': month_df["end station id"],
                'start_lat': month_df["start station latitude"],
                'start_lng': month_df["start station longitude"],
                'end_lat': month_df["end station latitude"],
                'end_lng': month_df["end station longitude"],
            })
        return month_df
    # else:
        # print(f"Month {year}-{month} missing.")


def load_year(year=2021):
    dataframes = []
    for i in range(1, 13):
        month_df = load_month(year, i)
        if month_df is not None and 'started_at' in month_df.columns:
            dataframes.append(month_df)
    return pd.concat(dataframes)


@st.cache(allow_output_mutation=True)
def load_data(year=2021, month_index=None):
    if month_index is None:
        df = load_year(year)
    else:
        df = load_month(year, month_index)

    # Preprocessing
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df.dropna(axis=0, inplace=True)

    # Remove invalid stations
    df = df.loc[df.start_station_id.str.contains('[a-zA-Z]')]
    df = df.loc[df.end_station_id.str.contains('[a-zA-Z]')]
    df = df.loc[~df.end_station_id.str.startswith('SYS')]

    # Compute distance and travel time
    df['distance'] = cal_distance(df.start_lng, df.end_lng, df.start_lat, df.end_lat)
    df['diff_time'] = (df['ended_at'] - df['started_at']).values / np.timedelta64(1, 'h') * 60

    return df, node_positions(df)


st.write("## Yearly Analysis")

year_sel = st.slider("Year", 2021, 2022, 2021)

data, node_position_df = load_data(year_sel, None)
possible_months = set(data.started_at.dt.month.to_list())

# st.write(data)
# st.write(node_position_df)


@st.cache(allow_output_mutation=True)
def plot_traffic_graph(g):

    chart = nxa.draw_networkx(
        G=g,
        pos=g.nodes.data('pos'),
        node_color='area',
        edge_color='black',
    )

    edges = chart.layer[0]
    nodes = chart.layer[1]

    brush = alt.selection_interval()
    color = alt.Color('area:N', legend=None)

    edges = edges.encode(
        opacity=alt.value(100 / len(g.edges)),
    )

    nodes = nodes.encode(
        opacity=alt.value(0.8),
        fill=alt.condition(brush, color, alt.value('gray')),
        size='traffic:Q',
        tooltip=[
            alt.Tooltip('name', title='Node name'),
            alt.Tooltip('lng', title='Longitude'),
            alt.Tooltip('lat', title='Latitude'),
            alt.Tooltip('traffic', title='Traffic')
        ]
    ).add_selection(
        brush
    )

    en_chart = (edges + nodes).properties(width=600)

    # Create a bar graph to show highlighted nodes.
    bars = alt.Chart(nodes.data).mark_bar().encode(
        x=alt.X('sum(traffic)', title='Total traffic'),
        y='area:N',
        color='area:N',
        tooltip=[
            alt.Tooltip('area', title='Area'),
            alt.Tooltip('count()', title='Number of nodes'),
            alt.Tooltip('sum(traffic)', title='Total traffic'),
        ]
    ).transform_filter(
        brush
    ).properties(width=600)

    return alt.vconcat(en_chart, bars)


year_graph = bike_network(data, node_position_df=node_position_df)
st.altair_chart(plot_traffic_graph(year_graph))


possible_color_strategies = ["PageRank", "Betweenness Centrality", "Closeness Centrality", "Eigenvector"]
color_strategy_sel = st.sidebar.selectbox("Coloring strategy", possible_color_strategies, 0)


@st.cache(allow_output_mutation=True)
def plot_graph(g, color_strategy="PageRank"):

    if color_strategy == "PageRank":
        node_colors = nx.algorithms.link_analysis.pagerank_alg.pagerank(g, weight="edge_traffic")
    elif color_strategy == "Betweenness Centrality":
        node_colors = nx.betweenness_centrality(g, weight="distance")
    elif color_strategy == "Closeness Centrality":
        node_colors = nx.closeness_centrality(g, distance='duration')
    elif color_strategy == "Eigenvector":
        node_colors = nx.eigenvector_centrality(g, max_iter=6000, weight="edge_traffic")
    else:
        node_colors = {n: 1.0 for n in g.nodes}
    for node in node_colors.keys():
        g.nodes[node]['color'] = node_colors[node]

    chart = nxa.draw_networkx(
        G=g,
        pos=g.nodes.data('pos'),
        node_color='color',
        edge_color='black',
        cmap='viridis',
    )

    edges = chart.layer[0]
    nodes = chart.layer[1]

    nodes = nodes.encode(
        x=alt.X('lng_scaled:Q', scale=alt.Scale(domain=(-2.7, 1.5))),
        y=alt.Y('lat_scaled:Q', scale=alt.Scale(domain=(-2.0, 2.0))),
        opacity=alt.value(0.8),
        size=alt.Size('traffic:Q', scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('name', title='Node name'),
            alt.Tooltip('lng', title='Longitude'),
            alt.Tooltip('lat', title='Latitude'),
            alt.Tooltip('traffic', title='Traffic'),
            alt.Tooltip('color', title=color_strategy)
        ]
    )

    edges = edges.encode(
        opacity=alt.value(100 / len(g.edges)),
    )

    return (edges + nodes).properties(width=800, height=500)


st.write("## Monthly Analysis (dynamic graph)")

month_graphs, month_data_lengths = {}, {}
for month in possible_months:
    month_data = data[data.started_at.dt.month == month]
    month_graphs[month] = bike_network(month_data, node_position_df=node_position_df)
    month_data_lengths[month] = format_number(len(month_data))

month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
month_names_inv = {v: k for k, v in month_names.items()}

if len(possible_months) > 1:
    month_sel = month_names_inv[st.select_slider(
        "Month", [month_names[month] for month in list(possible_months)])]
else:
    month_sel = min(possible_months)

text = st.empty()
text.markdown("Month: {} ({} data points)".format(
    month_names[month_sel], month_data_lengths[month_sel]))

month_chart = st.altair_chart(plot_graph(month_graphs[month_sel], color_strategy=color_strategy_sel))

# Add start and stop button and animate the chart over the months
col1, col2 = st.columns(2)

with col1:
    start = st.button("Start")
with col2:
    stop = st.button("Stop")

if start:
    for month in possible_months:
        text.markdown("Month: {}".format(month_names[month]))
        st.session_state['month'] = month
        month_chart.altair_chart(plot_graph(month_graphs[month], color_strategy=color_strategy_sel))
        time.sleep(1.0)

if stop and 'month' in st.session_state:
    month_chart.altair_chart(plot_graph(month_graphs[st.session_state['month']], color_strategy=color_strategy_sel))


st.write("## Weekly Analysis")

weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
weekday_names_inv = {v: k for k, v in weekday_names.items()}

last_months_data = data[data.started_at.dt.month.isin(list(possible_months)[-3:])]
st.write("Limited to the following months:", ", ".join([month_names[month] for month in list(possible_months)[-3:]]))

weekday_graphs, weekday_data_lengths = {}, {}
for weekday in weekday_names.keys():
    weekday_data = last_months_data[last_months_data.started_at.dt.dayofweek == weekday]
    weekday_graphs[weekday] = bike_network(weekday_data, node_position_df=node_position_df)
    weekday_data_lengths[weekday] = format_number(len(weekday_data))

weekday_sel = weekday_names_inv[st.select_slider("Day of the week", weekday_names.values())]
text = st.empty()
text.markdown("Day of the week: {} ({} data points)".format(
    weekday_names[weekday_sel], weekday_data_lengths[weekday_sel]))

st.altair_chart(plot_graph(weekday_graphs[weekday_sel], color_strategy=color_strategy_sel))


st.write("## Daily Analysis")

hour_graphs, hour_data_lengths = {}, {}
for hour in range(0, 24):
    hour_data = last_months_data[last_months_data.started_at.dt.hour == hour]
    hour_graphs[hour] = bike_network(hour_data, node_position_df=node_position_df)
    hour_data_lengths[hour] = format_number(len(hour_data))

hour_sel = st.slider("Hour of the day", 0, 23, 12)
text = st.empty()
text.markdown("Hour of the day: {}h ({} data points)".format(
    hour_sel, hour_data_lengths[hour_sel]))

st.altair_chart(plot_graph(hour_graphs[hour_sel], color_strategy=color_strategy_sel))


