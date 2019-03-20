#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from subprocess import check_output
print(check_output(["ls", "./data"]).decode("utf8"))

data_dir = './data/'
df_seeds = pd.read_csv(data_dir + 'Stage2DataFiles/NCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'Stage2DataFiles/NCAATourneyCompactResults.csv')


def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds.head()

df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tour.head()

df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed

df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_concat['SeedDiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))

X_train = df_predictions.SeedDiff.values.reshape(-1,1)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)


mlp = MLPClassifier()

mlp_parameters = {'max_iter': [1000,2000,10000], 'hidden_layer_sizes': [[100, 50, 25], [50, 25], [25]]}

clf = GridSearchCV(mlp, mlp_parameters, scoring='neg_log_loss', refit=True, cv=6, verbose=3 )

clf.fit(X_train, y_train)

print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))

X = np.arange(-10, 10).reshape(-1, 1)
preds = clf.predict_proba(X)[:,1]

plt.plot(X, preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')

mlp = MLPClassifier()
mlp_parameters = {'max_iter': [1750, 2000], 'hidden_layer_sizes': [[400, 200, 100], [200, 100, 50], [100, 50, 25]]}
clf = GridSearchCV(mlp, mlp_parameters, scoring='neg_log_loss', refit=True, cv=10, verbose=3 )

clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))

df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage2.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed


preds = clf.predict_proba(X_test)[:,1]

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub.head()

df_sample_sub.to_csv('submissions/ex2_mlp4.csv', index=False)




