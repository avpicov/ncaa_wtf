import glob
import pandas as pd
import numpy as np

path = './data/playbyplay/' # use your path
player_all_files = glob.glob(path + "PlayByPlay*/Players_*.csv")
event_all_files = glob.glob(path + "PlayByPlay*/Events_*.csv")

player_list = []

for filename in player_all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    player_list.append(df)

player_frame = pd.concat(player_list, axis=0, ignore_index=True)

event_list = []

for filename in event_all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    event_list.append(df)

event_frame = pd.concat(event_list, axis=0, ignore_index=True)


player_frame.head()

event_frame.head()

event_frame.describe()

seasons = event_frame.Season.unique()
teams = event_frame.EventTeamID.unique()
event_types = event_frame.EventType.unique()

event_type_group = event_frame.groupby(['EventTeamID', 'Season', 'EventType']).size()

event_season_group = event_frame.groupby(['EventTeamID', 'Season']).size()

stats_array = np.empty((len(event_type_group.keys().shape[0]), 3 + len(event_types)))

items = []
for key in event_season_group.keys():

    l = []
    l.append(key[0])
    l.append(key[1])
    for event_type in event_types:
        lookup_key = (key[0], key[1], event_type)
        if lookup_key in event_type_group.index:
            stat = event_type_group[lookup_key]

            l.append(stat)
        else:
            l.append(None)

    items.append(l)
    print(l)

stats = pd.DataFrame(items)

stat_columns = ['teamId', 'year']
for item in event_types.tolist():
    stat_columns.append(item)

stats.to_csv("stats.csv", index=False)