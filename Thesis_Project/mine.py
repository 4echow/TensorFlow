#!/usr/bin/env python
# coding: utf-8

"""
This code is a Keras implementation (only for tensorflow backend) of MINE: Mutual Information Neural Estimation (https://arxiv.org/pdf/1801.04062.pdf)
Author: Chengzhang Zhu
Email: kevin.zhu.china@gmail.com
Date: 2019-08-16
"""

"""

 Hyperparameter adjustments:
 
 We have edited the hyperparameters and model structure provided by C. Zhu [3] to make the algorithm applicable to our specific use case.
 Initially, we referred to the original MINE paper by Belghazi et al. 2018 for the statistics network architecture [2].
 Additionally, for guidance on setting hyperparameters for the MNIST experiment, we followed the referral by the team towards
 the settings described in the paper by Alemi et al. 2019 [1].
 The adjustments made here consider our dataset and objectives.
 For further details consult the readme_fs.txt file.
 
 Sources:
[1] A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy, “Deep Variational Information Bottleneck.” arXiv, Oct. 23, 2019. doi: 10.48550/arXiv.1612.00410.
[2] M. I. Belghazi et al., “Mutual Information Neural Estimation,” in Proceedings of the 35th International Conference on Machine Learning,  PMLR, Jul. 2018, pp. 531–540. Accessed: May 26, 2023. [Online]. Available: https://proceedings.mlr.press/v80/belghazi18a.html
[3] C. Zhu, “GitHub - ChengzhangZhu/MINE: Keras implementation (only for tensorflow backend) of MINE: Mutual Information Neural Estimation.” https://github.com/ChengzhangZhu/MINE (accessed Jun. 02, 2023).


 
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda, GaussianNoise
import keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001, # initial learning rate described by Alemi et al. 2019, among other hyperparams ([1], p.17) 
    decay_steps=1100, # with a batch_size=100 we have 550 batches per epoch, schedule applies every 2 epochs (s.a.)
    decay_rate=0.97, # the decay rate explained by Alemi et al. 2019 (s.a.)
    staircase=False) # Alemi et al. make no mention of staircase 

def mine_loss(args):
    t_xy = args[0]
    t_xy_bar = args[1]
    loss = -(K.mean(t_xy) - K.logsumexp(t_xy_bar) + K.log(tf.cast(K.shape(t_xy)[0], tf.float32)))
    return loss


def shuffle(y):
    return tf.random.shuffle(y)


class MINE(object):
    def __init__(self, x_dim=None, y_dim=None, network=None):
        self.model = None
        if network is None:
            assert x_dim is not None and y_dim is not None, 'x_dim and y_dim should be both given.'
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.network = self._build_network()
        else:
            assert isinstance(network, Model), 'the network should be defined as a Keras Model class'
            self.network = network

    def fit(self, x, y, epochs=200, batch_size=100, verbose=1):
        if self.model is None:
            self._build_mine()
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        else:
            assert len(y) == 1, 'only support that y is one target'
        inputs = x + y
        history = self.model.fit(x=inputs, epochs=epochs, batch_size=batch_size, verbose=verbose)
        fit_loss = history.history['loss']
        mutual_information = self.predict(x,y)
        return fit_loss, mutual_information

    def predict(self, x, y):
        assert self.model is not None, 'should fit model firstly'
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        else:
            assert len(y) == 1, 'only support that y is one target'
        inputs = x + y
        return np.mean(self.model.predict(x=inputs))

    def _build_mine(self):
        # construct MINE model
        x_input = self.network.inputs[0:-1]  # enable a complex x input
        y_input = self.network.inputs[-1]  # the last position in the input list should be y
        y_bar_input = Lambda(shuffle)(y_input)  # shuffle y input as y_bar
        t_xy = self.network(x_input + [y_input])
        t_xy_bar = self.network(x_input + [y_bar_input])
        loss = Lambda(mine_loss, name='mine_loss')([t_xy, t_xy_bar])
        output = Lambda(lambda x: -x)(loss)
        self.model = Model(inputs=x_input + [y_input], outputs=output, name='MINE_model')
        self.model.add_loss(loss)
        # we change beta_1 and beta_2 values to the ones described by Alemi et al. ([1], p. 17)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler, beta_1=0.5, beta_2=0.999))

    def _build_network(self):
        # 512-512-1 statistics network as described by Belghazi et al. [2].
        x = Input(shape=(self.x_dim,), name='network/x_input')
        y = Input(shape=(self.y_dim,), name='network/y_input')
        hidden = Concatenate(name='network/concatenate_layer')([x,y])
        hidden = GaussianNoise(0.3)(hidden) # add gaussian noise as data augmentation, following instructions from ([2], Table 15)
        for i in range(2): # nr. of hidden layers, 512 neurons each with gaussian noise layers (s.a.)
            hidden = Dense(512, activation='elu', name='network/hidden_layer_{}'.format(i+1))(hidden) 
            hidden = GaussianNoise(0.5)(hidden)
        output = Dense(1)(hidden)
        model = Model(inputs=[x, y], outputs=output, name='statistics_network')
        return model
