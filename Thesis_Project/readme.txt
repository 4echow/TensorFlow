"""

Full scale Experiment setup with a Lenet-300-100 base model for Lottery Tickets and a mine-512-512-1 statistics network.

Hyperparameters within Experiment_100.py and Initializations.py:

    # The fully connected model hyperparameters and architecture are based on the experimental setup by Frankle and Carbin, 2019 [3].
    # The team chose Lenet-300-100 architecture as standard ([3], p. 28, Fig. 31)
    
    input_dim = 784 # dim. size of MNIST input
    d1_dim = 300 # first hidden layer size for lottery ticket model 
    d2_dim = 100  # second hidden layer size for lottery ticket model
    o_dim = 10 # output layer size for lottery ticket model
    batch_size_LT = 60 # team uses mini-batches of size 60 for the MNIST experiment ([3], p. 24) 
    
    # number of epochs the lottery ticket model is trained, early-stop reached below 5000 training iterations ([3], Fig. 3, Fig. 31)
    # at minibatch size of 60 with 55000 training images => 917 iterations per full training cycle => ~5.45 epochs, we take 6 to round up
    epochs_LT = 6
    
    learning_rate_LT = 0.0012 # standard lr selected by Frankle et al. for Adam optimizer ([3], p. 26, Fig. 26)
    
    # we choose a final sparsity of 96.5%, the model with 3.5% remaining weights is the smallest among the still high performant tickets ([3], Fig. 12)
    pruning_rate = 0.2 # removes set percentage of lowest magnitude weights in a training cycle, rate set based on Frankle et al.([3], p.24)
    pruning_iterations = 15  # nr. of iterative pruning cycles -> 1 time : 20% sparse, 15 times : ~96,5% sparse model
    
    # Frankle et al. represent their winning tickets as averages of 5 trials, so we follow suit ([3], Fig. 1)
    averaging_iterations = 5 # number of total experimental runs to average for graph representations
    # The Experimental script has the number of averaging iterations set to 1, we run each iteration seperately on cluster pc's and concatenate results postscript

    
    # The lottery_ticket_pruner algorithm for producing lottery ticket networks was made by Jim Meyer [4]
    # we pip install it into our virtual environment and use it with above pruning rate on the described Lenet model
    
    # we use certain hyperparameters in Experiment_full_scale.py to set-up our MINE algorithm from mine.py:
    #batch size and epochs for mine are defined in ([1], p.17)
    batch_size_mine = 100 # defines the batch size for training the mine algorithm on our activation distributions
    epochs_mine = 100 # number of epochs we train the mine algorithm, source from [1] advices 200 epochs, however, we have memory constraints and need to take a smaller amount

Hyperparameters within mine.py:

    # The mine algorithm for measuring mutual information between neural network activation distributions was made by C. Zhu [5]
    # We download a copy of the script as we need to edit it to suit our usecase
    
    Changes made to mine.py:
    # changed tf.random_shuffle to tf.random.shuffle in the shuffle() function, as the prior is a deprecated version
    # We changed the model architecture to be in line with the MNIST experiment model described by Belghazi et al. ([2], Table 15)
    # Changed the first and second hidden layer size to have 512 neurons wit ELU activations each
    # Added GaussianNoise(stddev=0.3) after the input layer and GaussianNoise(stddev=0.5) after each hidden layer as described by the team
    # GaussianNoise adds noise to the previous layers outputs, it's a form of data augmentation
    # output layer remains a single neuron with ELU activation
    # Belghazi et al. refer us to Alemi et al., 2019 [1] for further hyperparameter setting for the MNIST experiment setup of MINE.
    # Following the instructions in the paper ([1], p.17) we add an exponential decay learning rate scheduler
    # it starts at an initial learning rate and slows down exponentially as training progresses
    # we also changed beta_1=0.5 and beta_2=0.999 in the Adam optimizer of the model's fit function as described in the setup (s.a.)
    
    Params for the tf.keras.optimizers.schedules.ExponentialDecay function [6]:
    initial_learning_rate=0.0001, # initial learning rate described by Alemi et al. 2019 (s.a.) 
    decay_steps=1100, # with a batch_size=100 we have 550 batches per epoch (55000 training data), schedule applies every 2 epochs (s.a.)
    decay_rate=0.97, # the decay rate explained by Alemi et al. 2019 (s.a.)
    staircase=False # Alemi et al. make no mention of staircase, we assume a smooth decay setup
    
    GaussianNoise [7] settings:
    stddev = 0.3 # for the noise layer after the Input layer
    stddev = 0.5 # for the noise layers after the two hidden layers
    
    Changes to Adam optimizer [8]:
    beta_1 = 0.5 # rate defined in [1] and as per description in [8] it's the exponential decay rate for first moment estimates
    beta_2 = 0.999 # rate defined in [1], per description in [8] it's the exponential decay rate for second moment estimates
    
    


Sources:   
[1] A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy, “Deep Variational Information Bottleneck.” arXiv, Oct. 23, 2019. doi: 10.48550/arXiv.1612.00410.
[2] M. I. Belghazi et al., “Mutual Information Neural Estimation,” in Proceedings of the 35th International Conference on Machine Learning, PMLR, Jul. 2018, pp. 531–540. Accessed: May 26, 2023. [Online]. Available: https://proceedings.mlr.press/v80/belghazi18a.html
[3] J. Frankle and M. Carbin, “The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.” arXiv, Mar. 04, 2019. doi: 10.48550/arXiv.1912.05671.
[4] J. Meyer, “Lottery Ticket Pruner.” Aug. 03, 2023. Accessed: Aug. 14, 2023. [Online]. Available: https://github.com/jim-meyer/lottery_ticket_pruner
[5] C. Zhu, “GitHub - ChengzhangZhu/MINE: Keras implementation (only for tensorflow backend) of MINE: Mutual Information Neural Estimation.” https://github.com/ChengzhangZhu/MINE (accessed Jun. 02, 2023).

[6] K. Team, “Keras documentation: ExponentialDecay.” https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/ (accessed Aug. 27, 2023).
[7] K. Team, “Keras documentation: GaussianNoise layer.” https://keras.io/api/layers/regularization_layers/gaussian_noise/ (accessed Aug. 29, 2023).
[8] K. Team, “Keras documentation: Adam.” https://keras.io/api/optimizers/adam/ (accessed Sep. 05, 2023).





"""