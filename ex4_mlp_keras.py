#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

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


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=1, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))

    model.add(Dense(50, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,  kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

keras_clf = KerasClassifier(build_fn=baseline_model, verbose=5, epochs=10)

param_grid = {
    'batch_size': [40],
    'epochs': [20]
}


# Create the grid
clf = GridSearchCV(estimator = keras_clf,
    param_grid = param_grid,
    scoring='neg_log_loss',
    n_jobs=1,
    iid=False,
    verbose=6,
    cv=6)

clf.fit(X_train, y_train)

print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))

X = np.arange(-10, 10).reshape(-1, 1)
preds = clf.predict_proba(X)[:,1]

plt.plot(X, preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')


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

df_sample_sub.to_csv('submissions/ex4_keras.csv', index=False)




