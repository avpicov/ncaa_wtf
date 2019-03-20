import glob
import pandas as pd

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