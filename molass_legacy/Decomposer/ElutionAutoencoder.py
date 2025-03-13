# coding: utf-8
"""
    ElutionAutoencoder.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
import numpy as np 
import matplotlib.pyplot    as plt

DEBUG   = False

USE_LINEAR_ACTIVATION   = True
USE_DEEP_CODERS         = True

def plot_loss( history ):
    #plot our loss 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

class ElutionAutoencoder:
    def __init__( self, a_curve, array ):

        X   = array
        print( 'X.shape=', X.shape )

        num_factors = len( a_curve.peak_info )

        input_dim = X.shape[1]
        encoding_dim = num_factors
        input_img = Input(shape=(input_dim,))
        if USE_LINEAR_ACTIVATION:
            encoded = Dense(encoding_dim, activation='linear')(input_img)
            # encoded = Dense(encoding_dim, activation='linear', activity_regularizer=l1_reg)(input_img)
            # encoded = Dense(encoding_dim, activation='linear', activity_regularizer=regularizers.l1(10e-5))(input_img)
            decoded = Dense(input_dim, activation='linear')(encoded)
        else:
            encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
            # encoded = Dense(encoding_dim, activation='relu', activity_regularizer=l1_reg)(input_img)
            # encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(1e-4))(input_img)
            decoded = Dense(input_dim, activation='relu')(encoded)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        print(autoencoder.summary())

        history = autoencoder.fit(X, X,
                        epochs=100,
                        batch_size=16,
                        # batch_size=256,
                        shuffle=True,
                        validation_split=0.1,
                        verbose = 0)

        plot_loss( history )

        # use our encoded layer to encode the training input
        encoder = Model(input_img, encoded)
        print(encoder.summary())
        # encoded_input = Input(shape=(encoding_dim,))
        # decoder_layer = autoencoder.layers[-1]
        # decoder = Model(encoded_input, decoder_layer(encoded_input))
        encoded_data = encoder.predict(X)

        fig = plt.figure( figsize=(16,7) )
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for k in range(num_factors):
            ax1.plot(encoded_data[:,k])

        ax2.plot( encoded_data[:,0], encoded_data[:,1], 'o' )

        fig.tight_layout()
        plt.show()
