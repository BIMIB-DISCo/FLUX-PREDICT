from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import argparse

import os
import time

import warnings
warnings.filterwarnings('ignore')

from model import create_model
from utils import get_experiment_path,  write_pickle

exp_path = get_experiment_path("results")

X = pd.read_csv(f"data/X_full.zip")

y = pd.read_csv(f"data/y_full.zip")


#################
# PREPROCESSING #
#################

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Save train and test set
X_train.to_csv(f"{exp_path}/X_train.zip")
y_train.to_csv(f"{exp_path}/y_train.zip")
X_test.to_csv(f"{exp_path}/X_test.zip")
y_test.to_csv(f"{exp_path}/y_test.zip")


####################
# MODEL DEFINITION #
####################

model = KerasRegressor(
    build_fn=create_model,
    input_shape = X_train.shape[1],
    output_shape = y_train.shape[1],
    epochs=5000,
    batch_size=128,
    verbose=0)


#########################
# HYPERPARAMETERS SPACE #
#########################

models = [(200,200), (100,), (200,), (500,), (100,100), (100,100,100)]
parameter_space = {
    'layers': models,
    'activation': ['relu'],
    'optimizer': ['adam', 'sgd'],
    'dropout' : [True, False],
    'learning_rate': [0.001, 0.01],
    'decay' : [0.9], #0.96
    'decay_steps' : [10000] #5000,
}


###############
# GRID SEARCH #
###############

grid = GridSearchCV(estimator=model,
                    param_grid=parameter_space,
                    n_jobs=6,
                    cv=5,
                    refit='r2',
                    verbose=1,
                    scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'])

earlyStop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=200,
                          verbose=0,
                          mode='auto',
                          restore_best_weights=True)

grid.fit(X_train, y_train.values, callbacks=[earlyStop], validation_split=.1)


###########
# RESULTS #
###########

grid_dict = {k: v for k, v in grid.__dict__.items() if 'estimator' not in k}

write_pickle(grid_dict, f"{exp_path}/CVcomplete_results.pkl")

est = grid.best_estimator_
est.model.save(f"{exp_path}/CVcomplete_best_estimator.h5")