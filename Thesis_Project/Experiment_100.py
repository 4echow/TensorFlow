#!/usr/bin/env python
# coding: utf-8

"""
Experiment for measuring mutual information using the MINE algorithm
between activation distributions of sparse 'lottery ticket' networks.

MINE algorithm is based on the Mutual Information Neural Estimation paper by Belghazi et al. [1].
The implementation was provided by C. Zhu [4] and adapted to our usecase.
The Lottery Ticket Hypothesis was first introduced by Frankle and Carbin in their 2019 paper on the phenomenon [2].
The implementation of the lottery ticket pruner was provided in the library made by Jim Meyer [3].

Sources:

[1] M. I. Belghazi et al., “MINE: Mutual Information Neural Estimation.” arXiv, Aug. 14, 2021. Accessed: May 26, 2023. [Online]. Available: http://arxiv.org/abs/1801.04062
[2] J. Frankle and M. Carbin, “The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.” arXiv, Mar. 04, 2019. doi: 10.48550/arXiv.1912.05671.
[3] J. Meyer, “Lottery Ticket Pruner.” Aug. 03, 2023. Accessed: Aug. 14, 2023. [Online]. Available: https://github.com/jim-meyer/lottery_ticket_pruner
[4] C. Zhu, “GitHub - ChengzhangZhu/MINE: Keras implementation (only for tensorflow backend) of MINE: Mutual Information Neural Estimation.” https://github.com/ChengzhangZhu/MINE (accessed Jun. 02, 2023).

"""


# In[1]: importing necessary dependencies

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import lottery_ticket_pruner
from lottery_ticket_pruner import LotteryTicketPruner, PrunerCallback
from mine import MINE
import pickle




# In[2]: loading MNIST data for training


# Load the MNIST dataset using TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
y_train_1hot = keras.utils.to_categorical(y_train, num_classes=10) # need y_train in a 1-hot encoded array for mine


# In[3]: Hyperparameters for the experiment

epochs_LT = 6 # epochs for the tickets, 5.45 epochs for about 5000 iterations, which is early-stop iteration in Frankle et al. paper
batch_size_LT = 60 # mini-batch size for the tickets
batch_size_mine = 100 # batch size for MINE algorithm
epochs_mine = 100 # epochs for MINE algorithm
validation_split = 1/11 # 5000 val 55000 train data
input_dim = 784 # input_distribution dim. for MINE, also dim. size of MNIST input
d1_dim = 300 # first hidden layer size for lottery ticket model, also first hidden layer activation distribution dim. for MINE
d2_dim = 100  # second hidden layer size for lottery ticket model, also second hidden layer activation distribution dim. for MINE
o_dim = 10 # output layer size for lottery ticket model, also output layer distribution dim. for MINE
pruning_rate = 0.2 # pruning rate for LTH iterative Pruning -> removes pruning_rate% of lowest magnitude weights in an iteration
pruning_iterations = 15  # number of iterations for applying the pruning rate iteratively -> 1 time : 20% sparse, 24 times : ~99.5% sparse
averaging_iterations = 1 # Frankle et al. usually use average of 5 trials
# we train this script as singular runs and concatenate the 5 results later





# In[4]: functions used within the iterative experimental process

def get_mask(pruned_model):
    """
    Returns a binary mask with the positions of zero-values in the pruned model.

    Parameters:
    pruned_model (ndarray): A numpy array representing the pruned model.

    Returns:
    ndarray: A binary mask with the same shape as the model, where 1 indicates non-zero values and 0 indicates zero values.
    """
    mask = []
    for i in range(1,len(pruned_model.layers)):
        weights = np.array((pruned_model.layers[i].get_weights()[0] != 0)*1.0)
        biases = np.array((pruned_model.layers[i].get_weights()[1] != 0)*1.0)
        layer = [weights, biases]
        mask.append(layer)
    return mask

def set_model(init_model,pruned_model):
    """
    Inserts the values from the initial model into the non-zero positions of the pruned model based on the provided mask.

    Parameters:
    initial_model (ndarray): A numpy array representing the initial model with values.
    pruned_model (ndarray): A numpy array representing the pruned model with zero values.
    mask (ndarray): A binary mask indicating the positions of non-zero values in the pruned model.

    Returns:
    ndarray: A modified pruned model with values from the initial model inserted at the non-zero positions based on the mask.
    """
    mask = get_mask(pruned_model)
    for i in range(1,len(init_model.layers)):
        layer = []
        weights = init_model.layers[i].get_weights()[0]
        biases = init_model.layers[i].get_weights()[1]
        pruned_model.layers[i].set_weights([np.where(mask[i-1][0] == 0, 0, weights), biases])# pruning doesn't zero out biases
                                                                                             # so we just copy the init-biases


def get_mine(distributions, epochs=200, batch_size=100):
    """
    Calculate mutual information using Mutual Information Neural Estimation (MINE) for multiple pairs of distributions.

    This function trains MINE models for multiple pairs of activation distributions of pruned lottery ticket models 
    and returns fit loss histories and mutual information estimations for each pair.

    Parameters:
    distributions (list of ndarray): A list of three distributions as NumPy arrays: [distrib_x, distrib_d1, distrib_d2, distrib_o].
    epochs (int, optional): The number of training epochs for each MINE model. Default is 200.
    batch_size (int, optional): The batch size used during training. Default is 100.

    Returns:
    tuple: A tuple containing two lists:
        - fit_loss_histories (list of list): A list of fit loss histories for each MINE model.
        - mutual_informations (list of float): A list of mutual information values for each MINE model.
    """
    fit_loss_histories = []
    mutual_informations = []
    
    distrib_x = np.copy(x_train) # input distribution, deep copy for safety
    distrib_d1 = distributions[0] # first hidden layer activations distribution
    distrib_d2 = distributions[1] # second hidden layer activations distribution
    distrib_o = distributions[2] # output layer distribution -> predicted output
    distrib_y = np.copy(y_train_1hot) # real output distribution, deep copy for safety
    
    mine_x_d1 = MINE(x_dim=input_dim, y_dim=d1_dim)
    fit_loss_history_x_d1, mutual_info_x_d1 = mine_x_d1.fit(distrib_x, distrib_d1, epochs=epochs, batch_size=batch_size)
    fit_loss_histories.append(fit_loss_history_x_d1)
    mutual_informations.append(mutual_info_x_d1)
    
    mine_x_d2 = MINE(x_dim=input_dim, y_dim=d2_dim)
    fit_loss_history_x_d2, mutual_info_x_d2 = mine_x_d2.fit(distrib_x, distrib_d2, epochs=epochs, batch_size=batch_size)
    fit_loss_histories.append(fit_loss_history_x_d2)
    mutual_informations.append(mutual_info_x_d2)
    
    mine_x_o = MINE(x_dim=input_dim, y_dim=o_dim)
    fit_loss_history_x_o, mutual_info_x_o = mine_x_o.fit(distrib_x, distrib_o, epochs=epochs, batch_size=batch_size)
    fit_loss_histories.append(fit_loss_history_x_o)
    mutual_informations.append(mutual_info_x_o)
    
    mine_d1_y = MINE(x_dim=d1_dim, y_dim=o_dim)
    fit_loss_history_d1_y, mutual_info_d1_y = mine_d1_y.fit(distrib_d1, distrib_y, epochs=epochs, batch_size=batch_size)
    fit_loss_histories.append(fit_loss_history_d1_y)
    mutual_informations.append(mutual_info_d1_y)
    
    mine_d2_y = MINE(x_dim=d2_dim, y_dim=o_dim)
    fit_loss_history_d2_y, mutual_info_d2_y = mine_d2_y.fit(distrib_d2, distrib_y, epochs=epochs, batch_size=batch_size)
    fit_loss_histories.append(fit_loss_history_d2_y)
    mutual_informations.append(mutual_info_d2_y)
    
    mine_o_y = MINE(x_dim=o_dim, y_dim=o_dim)
    fit_loss_history_o_y, mutual_info_o_y = mine_o_y.fit(distrib_o, distrib_y, epochs=epochs, batch_size=batch_size)
    fit_loss_histories.append(fit_loss_history_o_y)
    mutual_informations.append(mutual_info_o_y)
    
    return fit_loss_histories, mutual_informations


# just a quick function to iteratively determine sparsity level given a certain number of pruning operations with a given rate
def calc_sparsity(iteration, pruning_rate):
    sparsity = 100 * (1 - pruning_rate) ** (iteration+1)
    return 100-sparsity


# takes a list of loss_scores and averages out the dimension of the number of experiments -> 10 runs > average loss of 10 runs
def calculate_average_loss(losses_list):
    avg_losses = np.array(losses_list)
    avg_losses = np.average(avg_losses, axis=0)

    return avg_losses


# In[4]: base model with same architecture as our initial and trained lottery ticked base models in Initializations.py for loading


tf.keras.backend.clear_session() # clearing backend right at start, just in case

inputs = keras.Input(shape=(input_dim,), name="digits") # Functional build of a 2-hidden layer fully connected MLP
x = layers.Dense(d1_dim, activation="ReLU", name="dense_1")(inputs) # methods made no mention of the activaton function specifically
x = layers.Dense(d2_dim, activation="ReLU", name="dense_2")(x) # ReLU is standard, as all available implementations seem to use it too
outputs = layers.Dense(o_dim, activation="softmax", name="predictions")(x)  # softmax activation for multi-class classification

base_model = keras.Model(inputs=inputs, outputs=outputs)
base_model.summary()


# loading the saved initialization
base_model.load_weights("init_weights_fs.h5")
init_model = keras.models.clone_model(base_model)
init_model.load_weights("init_weights_fs.h5")
init_weights = init_model.get_weights() # init weights for Lottery Ticket reset to initial weights


# In[6]:


# Append the data lists to the dictionary for each iteration
data_dict = {"MI_x_y": [],
             "MI_history_x_y": [],
             
             "accuracies": [],
             "losses": [],
             "MI_estimates": [],
             "MI_histories": [],
       
             "accuracies_init": [],
             "losses_init": [],
             "MI_estimates_init": [],
             "MI_histories_init": [],

             "accuracies_rand": [],
             "losses_rand": [],
             "MI_estimates_rand": [],
             "MI_histories_rand": [],
             
             "accuracies_rand_init": [],
             "losses_rand_init": [],
             "MI_estimates_rand_init": [],
             "MI_histories_rand_init": [],

             }

# We do mine estimation between input and output distributions outside the loop, since x_train and y_train_1hot are static
# this saves us unnecessary computations
distrib_x_for_x_y = np.copy(x_train) # deep copy because we don't know how mine will affect iterative steps when referencing same structure
distrib_y_for_x_y = np.copy(y_train_1hot) 
mine_x_y = MINE(x_dim=input_dim, y_dim=o_dim)
fit_loss_history_x_y, mutual_info_x_y = mine_x_y.fit(distrib_x_for_x_y, distrib_y_for_x_y, epochs=epochs_mine, batch_size=batch_size_mine)
data_dict["MI_history_x_y"].append(fit_loss_history_x_y)
data_dict["MI_x_y"].append(mutual_info_x_y)

for j in range(averaging_iterations):
    print("------------------------")
    print("------------------------")
    print("------------------------")
    print("Experimental run number: " + str(j+1))
    print("------------------------")
    print("------------------------")
    print("------------------------")
    
    pruner_LF = LotteryTicketPruner(init_model) # pruner for large-final schedule
    pruner_rand = LotteryTicketPruner(init_model) # pruner for random pruning schedule
    
    # collected outputs for evaluation
    accuracies = []
    losses = []
    MI_estimates = []
    MI_histories = []

    
    accuracies_init = []
    losses_init = []
    MI_estimates_init = []
    MI_histories_init = []
    
    accuracies_rand = []
    losses_rand = []
    MI_estimates_rand = []
    MI_histories_rand = []
    
    accuracies_rand_init = []
    losses_rand_init = []
    MI_estimates_rand_init = []
    MI_histories_rand_init = []


    # compiling model with training params
    model = keras.models.clone_model(base_model) # model used for large-magnitude pruning later
    model_rand = keras.models.clone_model(base_model) # model used for random pruning later
    model.load_weights("init_weights_fs.h5") # loading in same initial weights each pruning iteration
    model_rand.load_weights("init_weights_fs.h5") # s.a.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1.2e-3), # Adam optimizer, lr=0.0012
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(), # multi-class classification loss function
                  # List of metrics to monitor
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                 )
    model_rand.compile(optimizer=keras.optimizers.Adam(learning_rate=1.2e-3), # Adam optimizer, lr=0.0012
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(), # multi-class classification loss function
                  # List of metrics to monitor
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                 )
    pre_train_loss, pre_train_accuracy = model.evaluate(x_test, y_test)
    accuracies.append(pre_train_accuracy)
    accuracies_init.append(pre_train_accuracy)
    accuracies_rand.append(pre_train_accuracy)
    accuracies_rand_init.append(pre_train_accuracy) # we'll save time by appending to rand outputs because same before prune
    losses.append(pre_train_loss)
    losses_init.append(pre_train_loss)
    losses_rand.append(pre_train_loss)
    losses_rand_init.append(pre_train_loss) # same for losses, before pruning there's no structural difference yet
    
    layer_names = ['dense_1', 'dense_2', 'predictions'] 

    # Create a function to extract activations from chosen layers
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_func = K.function([model.input], outputs)

    # Get activations for the input data
    activations = activation_func([x_train])
    histories, informations = get_mine(activations, epochs=epochs_mine, batch_size=batch_size_mine)
    
    # save results, both models are equal at this point pre pruning step
    MI_estimates.append(informations)
    MI_estimates_init.append(informations)
    MI_estimates_rand.append(informations)
    MI_estimates_rand_init.append(informations)
    MI_histories.append(histories)    
    MI_histories_init.append(histories)
    MI_histories_rand.append(histories)
    MI_histories_rand_init.append(histories)
    
    # fully-connected trained
    model.load_weights("trained_weights_fs.h5") # LOADING pre-trained weights for reproducibility
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1.2e-3), # Adam optimizer, lr=0.0012
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(), # multi-class classification loss function
                  # List of metrics to monitor
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                 )
    model_rand.load_weights("trained_weights_fs.h5") # LOADING pre-trained weights for reproducibility
    model_rand.compile(optimizer=keras.optimizers.Adam(learning_rate=1.2e-3), # Adam optimizer, lr=0.0012
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(), # multi-class classification loss function
                  # List of metrics to monitor
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                  )
    trained_loss, trained_accuracy = model.evaluate(x_test, y_test)
    accuracies.append(trained_accuracy) 
    accuracies_init.append(pre_train_accuracy) # if we take fully-connected trained model and put init_weights, we get the init_model
    accuracies_rand.append(trained_accuracy) # no prune yet, so same
    accuracies_rand_init.append(pre_train_accuracy) # s.a. same logic

    losses.append(trained_loss) # s.a. 
    losses_init.append(pre_train_loss) # s.a.
    losses_rand.append(trained_loss) # s.a.
    losses_rand_init.append(pre_train_loss) #s.a.

    MI_estimates_init.append(informations) # appending early, because I reuse variable names
    MI_histories_init.append(histories)    # the logic is if we have the fully trained model and reset
    MI_estimates_rand_init.append(informations) # the weights to init_weights, we just get the init_model from above again
    MI_histories_rand_init.append(histories) # so FT(set to init) = init, same for rand=FT pre pruning

    layer_names = ['dense_1', 'dense_2', 'predictions'] 

    # Create a function to extract activations from chosen layers
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_func = K.function([model.input], outputs)

    # Get activations for the input data
    activations = activation_func([x_train])
    histories, informations = get_mine(activations, epochs=epochs_mine, batch_size=batch_size_mine)
    
    # save results
    MI_estimates.append(informations)
    MI_estimates_rand.append(informations) # pre-pruning model_rand=model
    MI_histories.append(histories)
    MI_histories_rand.append(histories)

    for i in range(pruning_iterations):
        pruner_LF.set_pretrained_weights(model) # pruner for large_final pruning schedule
        pruner_rand.set_pretrained_weights(model_rand) # pruner for random pruning schedule
        model.set_weights(init_weights)
        model_rand.set_weights(init_weights)
        pruner_LF.calc_prune_mask(model, pruning_rate,'large_final') # pruner mask calculation based on large final magnitude
        pruner_rand.calc_prune_mask(model_rand, pruning_rate, 'random') # and random weight pruning schedules respectively
        sparsity = calc_sparsity(i,pruning_rate)
        print(f"Iteration {i+1}: making {sparsity:.2f}% sparse large_final")
        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size_LT,
                            epochs=epochs_LT,
                            shuffle=True,
                            verbose=0,
                            # monitoring validation loss and metrics
                            # at the end of each epoch
                            validation_data=(x_val, y_val),
                            callbacks=[PrunerCallback(pruner_LF)]) # making sure model trains with large_final schedule
        print("model, pruned LF: ")
        print(model.layers[3].get_weights()[0][:10])
        print(f"Iteration {i+1}: making {sparsity:.2f}% sparse random")
        history = model_rand.fit(x_train,
                             y_train,
                             batch_size=batch_size_LT,
                             epochs=epochs_LT,
                             shuffle=True,
                             verbose=0,
                             # monitoring validation loss and metrics
                             # at the end of each epoch
                             validation_data=(x_val, y_val),
                             callbacks=[PrunerCallback(pruner_rand)]) # making sure model_rand trains with random schedule
        print("model, pruned random: ")
        print(model_rand.layers[3].get_weights()[0][:10])
        print("")
        print("")
        ticket_loss, ticket_accuracy = model.evaluate(x_test, y_test)
        print(f"{sparsity:.2f}% sparse large_final: " + "loss: " + str(ticket_loss) + " acc: " + str(ticket_accuracy))
        model_init = keras.models.clone_model(model) # clone model for architecture
        model_init.set_weights(model.get_weights()) # put model weights into model_init so we have copy of model
        print("model_init, clone of model: ")
        print(model_init.layers[3].get_weights()[0][:10])
        set_model(init_model, model_init) # gives us model_init with non-zero weights set to initial weights and zero mask as in model
        print("init_model weights: ")
        print(init_model.layers[3].get_weights()[0][:10])
        print("model_init, init_weights set: ")
        print(model_init.layers[3].get_weights()[0][:10])
        model_init.compile(optimizer=keras.optimizers.Adam(learning_rate=1.2e-3), # Adam optimizer, lr=0.0012
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(), # multi-class classification loss function
                  # List of metrics to monitor
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                  )# using my set_function to take the ticket and manually set non-zero weights to init_weights
        ticket_loss_init, ticket_accuracy_init = model_init.evaluate(x_test, y_test)
        print(f"{sparsity:.2f}% sparse large_final, init_weights: " + "loss: " 
              + str(ticket_loss_init) + " acc: " + str(ticket_accuracy_init))
        random_loss, random_accuracy = model_rand.evaluate(x_test, y_test)
        print(f"{sparsity:.2f}% sparse random: " + "loss: " + str(random_loss) + " acc: " + str(random_accuracy))
        model_rand_init = keras.models.clone_model(model_rand) # clone model_rand for architecture
        model_rand_init.set_weights(model_rand.get_weights())  # put model_rand weights into model_rand_init
        print("model_rand_init, clone of model_rand: ")
        print(model_rand_init.layers[3].get_weights()[0][:10])
        set_model(init_model, model_rand_init) # s.a., gives us the sparse model with weights reset to initial weights if non-zero
        print("init_model weights: ")
        print(init_model.layers[3].get_weights()[0][:10])
        print("model_rand_init, init_weights set: ")
        print(model_rand_init.layers[3].get_weights()[0][:10])
        model_rand_init.compile(optimizer=keras.optimizers.Adam(learning_rate=1.2e-3), # Adam optimizer, lr=0.0012
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(), # multi-class classification loss function
                  # List of metrics to monitor
                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                  )# using my set_function to take the ticket and manually set non-zero weights to init_weights
        random_loss_init, random_accuracy_init = model_rand_init.evaluate(x_test, y_test)
        print(f"{sparsity:.2f}% sparse random, init_weights: " + "loss: " 
              + str(random_loss_init) + " acc: " + str(random_accuracy_init))
        print("")
        print("")
        accuracies.append(ticket_accuracy)
        accuracies_init.append(ticket_accuracy_init)
        accuracies_rand.append(random_accuracy)
        accuracies_rand_init.append(random_accuracy_init)

        losses.append(ticket_loss)
        losses_init.append(ticket_loss_init)
        losses_rand.append(random_loss)
        losses_rand_init.append(random_loss_init)

        layer_names = ['dense_1', 'dense_2', 'predictions'] 

        # Create functions via keras backend to get layer activations for the input data given, these will be our distributions for mine
        outputs = [model.get_layer(name).output for name in layer_names] 
        outputs_init = [model_init.get_layer(name).output for name in layer_names] 
        outputs_rand = [model_rand.get_layer(name).output for name in layer_names] 
        outputs_rand_init = [model_rand_init.get_layer(name).output for name in layer_names] 
        activation_func = K.function([model.input], outputs)
        activation_func_init = K.function([model_init.input], outputs_init)
        activation_func_rand = K.function([model_rand.input], outputs_rand)
        activation_func_rand_init = K.function([model_rand_init.input], outputs_rand_init)


        # Get activations for the input data
        activations = activation_func([x_train]) # list of layer activations from input data for model trained on large_final schedule
        activations_init = activation_func_init([x_train]) # s.a., except it's model_init with the non-zero values set to initial weights
        activations_rand = activation_func_rand([x_train]) # s.a. but with model_rand
        activations_rand_init = activation_func_rand_init([x_train]) # s.a., model_rand_init

        # apply the mine algorithm to the activation distributions
        histories, informations = get_mine(activations, epochs=epochs_mine, batch_size=batch_size_mine)
        histories_init, informations_init = get_mine(activations_init, epochs=epochs_mine, batch_size=batch_size_mine)
        histories_rand, informations_rand = get_mine(activations_rand, epochs=epochs_mine, batch_size=batch_size_mine)
        histories_rand_init, informations_rand_init = get_mine(activations_rand_init, epochs=epochs_mine, batch_size=batch_size_mine)
        
        
        MI_estimates.append(informations)
        MI_estimates_init.append(informations_init)
        MI_estimates_rand.append(informations_rand)
        MI_estimates_rand_init.append(informations_rand_init)
        
        MI_histories.append(histories)
        MI_histories_init.append(histories_init)
        MI_histories_rand.append(histories_rand)
        MI_histories_rand_init.append(histories_rand_init)
        
        
    # saving data for averaging     
    data_dict["accuracies"].append(accuracies)
    data_dict["losses"].append(losses)
    data_dict["MI_estimates"].append(MI_estimates)
    data_dict["MI_histories"].append(MI_histories)

    data_dict["accuracies_init"].append(accuracies_init)
    data_dict["losses_init"].append(losses_init)
    data_dict["MI_estimates_init"].append(MI_estimates_init)
    data_dict["MI_histories_init"].append(MI_histories_init)

    data_dict["accuracies_rand"].append(accuracies_rand)
    data_dict["losses_rand"].append(losses_rand)
    data_dict["MI_estimates_rand"].append(MI_estimates_rand)
    data_dict["MI_histories_rand"].append(MI_histories_rand)
    
    data_dict["accuracies_rand_init"].append(accuracies_rand_init)
    data_dict["losses_rand_init"].append(losses_rand_init)
    data_dict["MI_estimates_rand_init"].append(MI_estimates_rand_init)
    data_dict["MI_histories_rand_init"].append(MI_histories_rand_init)
    
with open(sys.argv[1],"wb") as file:
    pickle.dump(data_dict,file) # saving experimental results as pickled dictionary


