import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import backend as K

import time

def r2_metric(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_model(input_shape=10,
                 output_shape=10,
                 layers = [50],
                 activation = 'relu',
                 optimizer = 'adam',
                 learning_rate = 0.001,
                 dropout = True,
                 decay = 0.9,
                 decay_steps = 10000,
                 batch_norm=True):

    model = Sequential(name = str(time.time()))

    model.add(Input(input_shape))

    for l in layers:
        model.add(Dense(l))

        if batch_norm:
            model.add(BatchNormalization())

        model.add(Activation(activation=activation))

        if dropout:
            model.add(Dropout(0.5))

    model.add(Dense(output_shape))

    # Compile model
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate,
                                   decay_steps=decay_steps,
                                   decay_rate=decay)

    opt = optimizers.get({
        'class_name': optimizer,
        'config': {'learning_rate' : lr_schedule}})

    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=[r2_metric])

    return model