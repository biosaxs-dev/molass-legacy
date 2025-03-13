# coding: utf-8
"""
    QuantileRegressionKeras.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K

def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

class Model:
    def __init__(self):
        self.model = model = Sequential()
        model.add(Dense(units=10, input_dim=1,activation='relu'))
        model.add(Dense(units=10, input_dim=1,activation='relu'))
        model.add(Dense(1))
        # model.compile(loss='mae', optimizer='adadelta')
        model.compile(loss=lambda y,f: tilted_loss(0.5,y,f), optimizer='adadelta')

    def fit(self, x, y):
        self.model.fit(x, y, epochs=1000, batch_size=32, verbose=0)
        self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)
