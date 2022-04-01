
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageOps, UnidentifiedImageError


st.image("images/csgo.png")
st.sidebar.image("images/csgo_sidebar.png")

st.write("# Impact of individual performances on team performance in CS:GO")

st.markdown("*By GRANGER Dimitri, L'HOSTIS Brieg, LI Zhenning, WU Qiyun - March 2022*")

st.write("Counter-Strike Global Offensive (CS:GO) is an online First-Person Shooter with a significant fan base "
         "and presence within the esport scene for many years. The dataset applied in this dashboard includes "
         "information about each CS:GO professional match ranging from 2016 to 2019. This dashboard focuses on "
         "the individuals making up a team: how do the performance of each player during the games correlate with "
         "the performances of the team? By exploring the consistency in the ratings of the players and how the "
         "performances of players in top teams have changed through time, you can develop a comprehensive view "
         "of skillsets on both the players and the teams.")

# st.write("## 0. Load data")


def load_data():
    if os.path.isfile("players.csv"):
        return pd.read_csv("results.csv"), pd.read_csv("players.csv")
    players_sub_data = []
    for j in range(1, 4):
        players_sub_data.append(pd.read_csv("players_{}.csv".format(j)))
    return pd.read_csv("results.csv"), pd.concat(players_sub_data)


results_df, players_df = load_data()

# st.write("Results database")
# st.dataframe(results_df.head(100))

# st.write("Players database")
# st.dataframe(players_df.head(100))

# Filter on date
results_df["date"] = pd.to_datetime(results_df["date"], format="%Y-%m-%d")
players_df["date"] = pd.to_datetime(players_df["date"], format="%Y-%m-%d")
# Remove invalid data (teams with no name or no player)
results_df = results_df[results_df.team_1 != "?"]
results_df = results_df[results_df.team_2 != "?"]
players_df = players_df[players_df.team != "?"]
possible_teams = list(players_df.team.unique())
results_df = results_df[results_df.team_1.isin(possible_teams)]
results_df = results_df[results_df.team_2.isin(possible_teams)]

year_sel = st.sidebar.slider("Year", 2016, 2019, 2019)

results_source_df = results_df[results_df.date.dt.year == year_sel]
players_source_df = players_df[players_df.date.dt.year == year_sel]

st.write("## 1. Rating Consistency")
st.sidebar.write("## 1. Rating Consistency")

top_options = [5, 10, 20, 50, 100]
top_teams_sel = st.sidebar.selectbox("Number of top teams", top_options, 1)
top_players_sel = st.sidebar.selectbox("Number of top players", top_options, 3)

# Compute team rating
team1_df = pd.DataFrame({"team": results_source_df["team_1"], "rank": results_source_df["rank_1"],
                         "match_id": results_source_df["match_id"],
                         "win_rate": (results_source_df["match_winner"] == 1).astype(int)})
team2_df = pd.DataFrame({"team": results_source_df["team_2"], "rank": results_source_df["rank_2"],
                         "match_id": results_source_df["match_id"],
                         "win_rate": (results_source_df["match_winner"] == 2).astype(int)})
team_df_ungrouped = pd.concat((team1_df, team2_df))
team_df_ungrouped = team_df_ungrouped.drop_duplicates(["team", "match_id"]).drop("match_id", axis=1)
team_df = team_df_ungrouped.groupby("team", as_index=False).mean()
team_df["matches"] = team_df_ungrouped.groupby("team", as_index=False).count()["win_rate"]
# Filter the top teams
team_df = team_df.sort_values(by="rank", ascending=True)
team_df = team_df.head(max(top_options))

# Compute the player consistency coefficient
top_team_names = list(team_df.head(top_teams_sel).team.unique())
players_rating_df = players_source_df[players_source_df.team.isin(top_team_names)]
players_rating_df = players_rating_df[["player_name", "team", "rating"]]
players_rating_stats_df = players_rating_df.groupby(["player_name", "team"], as_index=False).mean()
players_rating_stats_df["count"] = players_rating_df.groupby(["player_name", "team"], as_index=False).count()["rating"]
# Filter the top players with at least one match
players_rating_stats_df = players_rating_stats_df[players_rating_stats_df["count"] > 1]
players_rating_stats_df = players_rating_stats_df.sort_values(by="rating", ascending=False)
players_rating_stats_df = players_rating_stats_df.head(top_players_sel)
# Compute std and consistency
players_rating_stats_df["std"] = players_rating_df.groupby(["player_name", "team"], as_index=False).std()["rating"]
players_rating_stats_df["consistency"] = (1 + 0.25/players_rating_stats_df["count"]) *\
                                         (players_rating_stats_df["std"] / players_rating_stats_df["rating"])
players_rating_stats_df["team_ranking"] = [team_df[team_df.team == team].iloc[0]["rank"] for
                                           team in players_rating_stats_df.team]

# st.write(players_rating_stats_df)


@st.cache(allow_output_mutation=True)
def bubble_chart(stats_df, bar_x_axis="rating", title=""):
    selection = alt.selection_interval()

    bubbles = alt.Chart(stats_df, title=alt.TitleParams(title, fontSize=20)).mark_circle(
        opacity=0.7, stroke='black', strokeWidth=1).encode(
        x=alt.X('rating:Q', axis=alt.Axis(title='Average player rating'), scale=alt.Scale(zero=False)),
        y=alt.Y('consistency:Q', axis=alt.Axis(title='Player consistency coefficient'), scale=alt.Scale(zero=False)),
        size=alt.Size('team_ranking:Q', title='Team average ranking', scale=alt.Scale(zero=False), sort="descending"),
        color=alt.condition(selection, alt.Color('team:N'), alt.value('lightgray'), title='Team name'),
        # strokeOpacity=alt.condition(f'datum.team == "{team_selection}"', alt.value(1.0), alt.value(0.0)),
        tooltip=[
            alt.Tooltip('team', title='Team name'),
            alt.Tooltip('player_name', title='Player name'),
            alt.Tooltip('team_ranking', title='Average team ranking'),
            alt.Tooltip('rating', title='Average player rating'),
            alt.Tooltip('consistency', title='Player consistency'),
        ]
    ).add_selection(selection).properties(width=600)

    x_rule = alt.Chart(stats_df).mark_rule().encode(
        x='mean(rating):Q',
        tooltip=[alt.Tooltip('mean(rating)', title='Mean player rating')],
    )
    y_rule = alt.Chart(stats_df).mark_rule().encode(
        y='mean(consistency):Q',
        tooltip=[alt.Tooltip('mean(consistency)', title='Mean player consistency')],
    )
    bubbles += x_rule + y_rule

    bars = alt.Chart(stats_df).mark_bar().encode(
        y=alt.Y('team:N', title='Team', sort=alt.EncodingSortField(field='team_ranking', order='ascending')),
        color=alt.Color('team:N', title='Team name'),
        x=alt.X('mean({}):Q'.format(bar_x_axis), title='Average {}'.format(bar_x_axis), scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('team', title='Team name'),
            alt.Tooltip('team_ranking', title='Average team ranking'),
            alt.Tooltip('mean(rating)', title='Average player rating'),
            alt.Tooltip('mean(consistency)', title='Average player consistency'),
        ]
    ).transform_filter(selection).properties(width=600)

    bars += bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='num_players:N'
    ).transform_joinaggregate(
        group_count='count()',
        groupby=['team']
    ).transform_calculate(
        num_players='datum.group_count + " players"',
    )

    return alt.vconcat(bubbles, bars)


x_axis_sel = st.sidebar.selectbox("X-axis on the bar chart", ["rating", "consistency"])
st.altair_chart(
    bubble_chart(players_rating_stats_df, bar_x_axis=x_axis_sel,
                 title="Player average rating vs. rating consistency in {}".format(year_sel)),
    use_container_width=True)
st.caption("The consistency coefficient is computed from the player's in-game ratings in the year. This coefficient "
           "aims at measuring the player's consistency in their rating. It is computed by dividing the standard "
           "deviation of the ratings by the mean of the ratings and corrected by a factor to take "
           "the number of matches played into account. The lines on the bubble chart show the average player "
           "consistency and rating to show which player are above average in consistency and/or rating.")
st.caption("The bar chart shows the average rating of the selected teams' players. The teams are sorted by "
           "their rank ascending from top to bottom. The number of players selected for each team is also "
           "displayed to give the user information about which teams are most represented in their "
           "selection.")


st.write("## 2. Performances of top teams and their players")
st.sidebar.write("## 2. Performances of top teams and their players")

st.write("### 2.1. Top teams")

# Find the top 5 teams
top = 25
top_teams = team_df.head(top)
top_teams["overall_rank"] = top_teams["rank"].rank().astype(int)
top_teams_players_df = players_source_df[players_source_df["team"].isin(list(top_teams.team.unique()))]
top_teams_players_df = top_teams_players_df[["date", "team", "player_name", "kills", "assists",
                                             "deaths", "hs", "flash_assists", "rating"]]
top_teams_player_stats_df = top_teams_players_df.drop(["date", "player_name"], axis=1)
top_teams_player_stats_df = top_teams_player_stats_df.groupby("team", as_index=False).mean()

# st.write(top_teams_players_df)

top_teams["player_rating"] = [top_teams_player_stats_df[top_teams_player_stats_df.team == team].iloc[0]["rating"]
                              for team in top_teams.team]
top_teams["kills"] = [top_teams_player_stats_df[top_teams_player_stats_df.team == team].iloc[0]["kills"]
                      for team in top_teams.team]
top_teams["assists"] = [top_teams_player_stats_df[top_teams_player_stats_df.team == team].iloc[0]["assists"]
                        for team in top_teams.team]
top_teams["deaths"] = [top_teams_player_stats_df[top_teams_player_stats_df.team == team].iloc[0]["deaths"]
                       for team in top_teams.team]
top_teams["hs"] = [top_teams_player_stats_df[top_teams_player_stats_df.team == team].iloc[0]["hs"]
                   for team in top_teams.team]
top_teams["flash_assists"] = [top_teams_player_stats_df[top_teams_player_stats_df.team == team].iloc[0]["flash_assists"]
                              for team in top_teams.team]

# st.dataframe(top_teams)


def load_and_resize(img_path, desired_size):
    img = Image.open(img_path)
    width, height = img.size
    if width > height:
        new_width, new_height = desired_size, int(height * desired_size / width)
    else:
        new_width, new_height = int(width * desired_size / height), desired_size
    img = img.resize((new_width, new_height))
    # Pad the image to have a square image
    delta_w = desired_size - new_width
    delta_h = desired_size - new_height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding)
    return img


SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd'}


def ordinal(num):
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = SUFFIXES.get(num % 10, 'th')
    return str(num) + suffix


top_teams_list = top_teams.team.to_list()
top_teams_ranking_list = ["{} (rank {})".format(team, rank)
                          for team, rank in zip(top_teams.team, top_teams.overall_rank)]
top_teams_ranking_list.sort()
team_sel = st.sidebar.selectbox("Team to display", top_teams_ranking_list,
                                index=top_teams_ranking_list.index(top_teams_list[0] + " (rank 1)")).split(" (rank ")[0]
cols = st.columns(5)
for i, col in enumerate(cols):
    rank = i+1
    team = top_teams_list[i]
    with col:
        if os.path.isfile("images/{}.png".format(team)):
            try:
                logo = load_and_resize("images/{}.png".format(team), 500)
                st.image(logo)
            except UnidentifiedImageError:
                st.write("We have issues here...")
        else:
            st.write("No image here!")
        st.write("{} - {}".format(ordinal(rank), team))
        if st.button("Select this team", key=team):
            team_sel = team


stat_names = ["Win rate", "Nb. matches", "Avg. player rating", "Avg. nb. kills",
              "Avg. nb. assists", "Avg. nb. deaths", "Avg. nb. headshots"]
stat_columns = ["win_rate", "matches", "player_rating", "kills", "assists", "deaths", "hs"]
stat_means = [top_teams[stat_col].mean() for stat_col in stat_columns]
stat_stds = [top_teams[stat_col].std() for stat_col in stat_columns]
team_stats = top_teams[top_teams.team == team_sel].iloc[0]
stat_values = [team_stats[stat_col] for stat_col in stat_columns]

st.write("### 2.2. Team performance - {} - Rank {}".format(team_sel, team_stats["overall_rank"]))

st.markdown("**Overall statistics compared to the top {} teams:**".format(top))


def change_color(val, reverse=False):
    if reverse:
        color = 'red' if val > 0 else 'blue'
    else:
        color = 'blue' if val > 0 else 'red'
    return f'color: {color}'


data_dict = {stat_name: [stat_values[i], (stat_values[i] - stat_means[i])/stat_stds[i]]
             for i, stat_name in enumerate(stat_names)}
table_data = pd.DataFrame(data_dict, index=["Value", "Normalized value"])
table_data = table_data.style.applymap(
    change_color, subset=("Normalized value", ["Win rate", "Avg. player rating", "Avg. nb. kills",
                                               "Avg. nb. assists", "Avg. nb. headshots"]))
table_data = table_data.applymap(change_color, reverse=True, subset=("Normalized value", "Avg. nb. deaths"))
st.table(table_data)
st.caption("The normalized value is computed by normalizing the team's statistics with respect to the "
           "top {} teams' statistics (with mean and standard deviation). On the bottom line, the color is "
           "changed to show whether the team's statistic is better (in blue) or worse (in red) than the average "
           "statistic for the top {} teams.".format(top, top))

st.markdown("**Performance evolution in {}:**".format(year_sel))

# Group the data by month
team_players_df = top_teams_players_df[top_teams_players_df["team"] == team_sel]
team_players_df = team_players_df[["date", "kills", "assists", "deaths", "hs", "flash_assists"]]
period = team_players_df.date.dt.to_period("M")
team_players_df = team_players_df.groupby(period, as_index=False).mean()
team_players_df["month"] = range(len(team_players_df))

team_players_df["kills/deaths"] = team_players_df.kills / team_players_df.deaths
team_players_df["kills and assists/deaths"] = (team_players_df.kills + team_players_df.assists) / team_players_df.deaths
team_players_df["headshots/kills"] = team_players_df.hs / team_players_df.kills


# Compute the team's win rate
team1_df_dated = pd.DataFrame({"team": results_source_df["team_1"], "date": results_source_df["date"],
                               "match_id": results_source_df["match_id"],
                               "win_rate": (results_source_df["match_winner"] == 1).astype(int)})
team2_df_dated = pd.DataFrame({"team": results_source_df["team_2"], "date": results_source_df["date"],
                               "match_id": results_source_df["match_id"],
                               "win_rate": (results_source_df["match_winner"] == 2).astype(int)})
team_df_dated = pd.concat((team1_df_dated, team2_df_dated))
team_df_dated = team_df_dated[team_df_dated.team == team_sel].drop("team", axis=1)
team_df_dated_ungrouped = team_df_dated.drop_duplicates("match_id").drop("match_id", axis=1)
period_df_dated = team_df_dated_ungrouped.date.dt.to_period("M")
team_players_df["win rate"] = team_df_dated_ungrouped.groupby(
    period_df_dated, as_index=False).mean()["win_rate"]
team_players_df["monthly matches/yearly matches"] = team_df_dated_ungrouped.groupby(
    period_df_dated, as_index=False).count()["win_rate"] / team_stats["matches"]


# Compute the label expression for the month names
month_names = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.",
               "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
month_full_names = ["January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December"]
month_labels = " ".join(["datum.label == {} ? '{}':".format(i, month_name) for i, month_name in enumerate(month_names)])
month_labels += "''"


# Line chart
@st.cache(allow_output_mutation=True)
def line_chart(chart_data, label_expression, title=""):
    hover = alt.selection_single(
        fields=["month"],
        nearest=True,
        on="mouseover",
        empty="none",
        clear="mouseout",
    )

    lines = (
        alt.Chart(chart_data, title=title).mark_line().encode(
            x=alt.X('month:Q', axis=alt.Axis(title='Month', labelExpr=label_expression, labelAngle=-45)),
            y=alt.Y('value:Q', axis=alt.Axis(title='Average statistic')),
            color='Statistics:N',
        )
    )

    selectors = alt.Chart(chart_data).mark_point().encode(
        x='month:Q',
        opacity=alt.value(0),
    ).add_selection(hover)

    points = lines.mark_point().encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )

    text = lines.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(hover, 'Value:N', alt.value(' '))
    )

    rules = alt.Chart(chart_data).mark_rule(color='gray').encode(
        x='month:Q',
    ).transform_filter(hover)

    return (lines + selectors + points + text + rules).properties(width=800, height=500)


data = team_players_df.drop(["kills", "assists", "deaths", "hs", "flash_assists"], axis=1)
data = data.melt('month', ignore_index=True)
data["Month"] = [month_full_names[i] for i in data["month"]]
data["Statistics"] = data["variable"].replace("hs", "headshots").replace("flash_assists", "flash assists")
data["Value"] = ["{:.2f} {}".format(data.iloc[i]["value"], data.iloc[i]["Statistics"]) for i in range(len(data))]
team_chart = st.altair_chart(
    line_chart(data, month_labels, title="Performances of team '{}' in {}".format(team_sel, year_sel)),
    use_container_width=False)
st.caption("In the graph above, the ratios including kills, assists, deaths or headshots are computed by "
           "taking the number of kills, assists, deaths or headshots per game averaged over the month. The "
           "'monthly matches/yearly matches' is simply the number of matches in the month divided by "
           "the number of matches in the year for this team.")


st.write("### 2.3. Player performances")

# Get the 5 most frequent players
team_players_df = top_teams_players_df[top_teams_players_df["team"] == team_sel]
players = team_players_df.player_name.value_counts()
players = players.sort_values(ascending=False)
players = players.head(5).index.tolist()

player_stats_df = team_players_df[["player_name", "kills", "assists", "deaths", "hs", "flash_assists", "rating"]]
player_stats_df = player_stats_df[player_stats_df.player_name.isin(players)]
player_stats_df = player_stats_df.groupby("player_name", as_index=False).mean()
player_stats_df = player_stats_df.sort_values("rating", ascending=False)
players = player_stats_df.player_name.to_list()


# Radar chart
@st.cache(allow_output_mutation=True)
def radar_chart(r, theta, size=8):
    r, theta = [*r, r[0]], [*theta, theta[0]]
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(theta))
    fig = plt.figure(figsize=(size, size))
    plt.subplot(polar=True)
    plt.plot(label_loc, r)
    plt.thetagrids(np.degrees(label_loc), labels=theta)
    return fig


cols = st.columns(5)
radar_labels = ["K", "A", "D", "HS", "FA"]
stats_columns = ["kills", "assists", "deaths", "hs", "flash_assists"]
normalization = [player_stats_df[stat_col].max() for stat_col in stats_columns]
for i, col in enumerate(cols):
    player = players[i]
    player_stats = player_stats_df[player_stats_df.player_name == player].iloc[0]
    with col:
        st.markdown(f"[**{player}**](https://liquipedia.net/counterstrike/{player})")
        data = [player_stats[stat_col]/normalization[i] for i, stat_col in enumerate(stats_columns)]
        st.pyplot(radar_chart(data, radar_labels, size=3))
        st.write("Rating: {:.3f}".format(player_stats["rating"]))
st.caption("In the radar charts, the labels 'K', 'A', 'D', 'HS' and 'FA' respectively represent the average "
           "number of kills, assists, deaths, headshots and flash-assists done by the player in matches over "
           "the year {}.".format(year_sel))
st.caption("The values displayed in the radar charts are normalized by dividing by the maximum value "
           "in the team. Note that the player's rating is a complex statistic computed by the game "
           "and thus is not comparable to the statistics displayed in the radar chart.")

