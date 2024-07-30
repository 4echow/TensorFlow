#!/usr/bin/env python
# coding: utf-8

# In[1]: import dependencies


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


# In[2]: load MNIST data and prepare for training


# Load the MNIST dataset using TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Display the shapes of the training and test datasets
print("Training data shape:", x_train.shape, y_train.shape)
print("Test data shape:", x_test.shape, y_test.shape)

# reshape data as 2D numpy arrays
# convert to float32 and normalize grayscale for better num. representation
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# The tutorial reserved 10.000 training samples for validation, we change to 5.000 
# as that is what Frankle and Carbin did in their paper
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]


# In[3]: Hyperparams

batch_size_LT = 60 # batch size for lottery ticket model selected as standard by Frankle et. al
epochs_LT = 6 # epochs for lottery ticket model, with early-stop at around 5000 iterations, with a batch size of 60 occurs at epoch=5.45
learning_rate_LT = 1.2e-3 # learning rate for Adam optimizer selected by Frankle et. al as standard
input_dim = 784 # dim. size of MNIST input
d1_dim = 300 # first hidden layer size for lottery ticket model
d2_dim = 100  # second hidden layer size for lottery ticket model
o_dim = 10 # output layer size for lottery ticket model


# In[4]: Initialize a fully connected model based on the Lenet-300-100 architecture described by Frankle and Carbin in their paper
# and save its initial weights.
# Train model and save trained weights
# This will be used for consistency during the iterative lottery ticket pruning process.

# Clear the backend session at the start.
tf.keras.backend.clear_session()

inputs = keras.Input(shape=(input_dim,), name="digits") # Functional build of a 2-hidden layer fully connected MLP
x = layers.Dense(d1_dim, activation="ReLU", name="dense_1")(inputs) # methods made no mention of the activaton function specifically
x = layers.Dense(d2_dim, activation="ReLU", name="dense_2")(x) # ReLU is standard, as all available implementations seem to use it too
outputs = layers.Dense(o_dim, activation="softmax", name="predictions")(x)  # softmax activation for multi-class classification

base_model = keras.Model(inputs=inputs, outputs=outputs)

# we save the initial weights for later use here
base_model.save_weights('init_weights_fs.h5') 

# train fully-connected model
model = keras.models.clone_model(base_model) # clones the model, same weights and architecture, just a precaution to not edit wrong models
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate_LT), # Adam optimizer, lr=0.0012
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(), # multi-class classification loss function
                  # List of metrics to monitor
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                 )


history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size_LT,
                    epochs=epochs_LT,
                    verbose=0,
                    shuffle=True,
                    validation_data=(x_val, y_val),
                    )

trained_loss, trained_accuracy = model.evaluate(x_test, y_test)


model.save_weights("trained_weights_fs.h5") # saving trained weights for later use
