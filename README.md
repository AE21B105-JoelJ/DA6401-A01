# DA6401 - Deep Learning Assignment 01 - (AE21B105)
This README file gives the information about the code and how the function space is defined and used. This is a assignment on the programming and testing of the back-propagation framework and different optimization algorithms. Now all beign said lets get into the gory details of the assignment (code).

## Links to report and project
- GITHUB Link : https://github.com/AE21B105-JoelJ/DA6401-A01/
- WandB link : https://api.wandb.ai/links/A1_DA6401_DL/ddwn0shz
## Usage of the script
To run the code use the following code (The external parameters are defaulted to best accuracy got!!!), all the modules and classes are present in source.py file. The single training can be done with the train.py file.

```
python train.py --wandb_project project_name --wandb_entity entity_name
```

the additional supported arguents are as follows
- "--dataset" tells dataset on which the model will be trained
- "--epochs" tells Number of epochs to train the model
- "--batch_size" tells batch size used to train the network
- "--loss" tells the loss function to use
- "--optimizer" tells optimizer to be used in training
- "--learning_rate" tells learning rate used in the gradient update
- "--momentum" tells momentum used in momentum-GD and NAG
- "--beta" tells Beta used by RMSprop
- "--beta1" tells Beta_1 used by Adam and NAdam
- "--beta2" tells Beta_2 used by Adan and NAdam
- "--epsilon" tells epsilon used by the optimizers
- "--weight_decay" tells Weight decay used in training (L2)
- "--weight_init" tells Weight initialization for the params
- "--num_layers" tells number of hidden layers
- "--hidden_size" tells number of neurons per hidden layer
- "--activation" tells activation to be used for the hidden layers

## Definition of the functions
 - The first function "init_mat()" is used to initiate the weights and biases of a neural network given the architecture and the intitialization scheme which is either random or Xavier. For both case the random number generator used is gaussian distribution for random I have used a standard deviation of 0.1 and for Xavier its as per the definition of it. Note here Weights and Biases are list of <numpy.ndarray> which are squentially used for forward propagation.

 - The next functions explained are which are "one_hot_numpy()", "find_loss()" and "accuracy()" as the name suggests is to find one-hot vectors from categotical arrays, find the loss given the prediction, true and the loss function and accuracy function is to find the accuracy.

 -  The next set of functions names as "relu()", "linear()", "tanh()", "sigmoid()", "softmax()" are used to find the activation given the input and the activation function.

 -  The next set of finctions names as "diff_relu()", "diff_linear()", "diff_tanh()", "diff_sigmoid()", "diff_softmax_idea()" are used to find the diffentiation of the functions output with respect to the inputs except the last one. "softmax_diff_idea()" is used to find the gradient of the outputs_logits with the predicted_logits using the MSE loss discussed in the report (A workaround hack for softmax + MSE combo).

## Main functions
Build of the neural network
 - arch = [784, 128, 10] means that Input is 784 dimensional first hidden layer has 128 neurons and output has 10 neurons
 - activation_sequence = [relu, softmax] means that the hidden layer activation is relu and output activation is softmax
### Forward Propagation
  The function "forward_propagation()" is used to progate through the neural network given the Weights, Biases, activation_sequence anf then forward propagation is done and returns the output, pre-activations of each layer, post-activations of each layer (to be used in back-propagation).
  Note : Here the training and testing data are of the shape (batch_size, dimension) but once we enter the forward-propagation and back-propagation we will always reshape the input and other matrices of the shape (dim, batch_size). The output matrix from the forward propagation is reshaped to match with the dataset type.

### Batchloader class
  The batch loader class is used to return the batches of dataset both features and labels in a batch-wise fashion by using the "__iter__" and "__next__" method of a class. Also, an option to shufffle the points is also added as a feature.

### Optimizer class
   The optimizer class is the main part of the assignment as it contains the back-propagation framework , optimization updater and a orchestrator to orchestrate this when called form a model class which we will discuss later.

 - "__init__" method is used to initialize the variables needed for the optimizations to happen such as loss, optimizer, learning_rate, momentum, beta_1, beta_2, weight_decay, etc
 - "backprop_grads()" function does the heavylifting part it calculates the gradients with respect to the Biases and Weights as a list of <numpy.ndarray> given the current weights, pre-activations, post-activations, y_true and activation_sequence. This function can be called under any optimizer_method to calculate the gradients at a timestep.
 - "gd_step()", "sgd_step()", "nag_step()", "rmsprop_step", "adam_step()", "nadam_step()" are the set of functions that has all the inputs necessary to call the "backprop_grads()" function, finds the gradient and makes an update to the existing parameters and retuns the updated Weights and Biases.
 - "stepper()" function is called from the model while training is used to call one of the above optimizer methods to do a one step gradient update and also keeps track of the number of updates done (used for adaptive models).

### Model (FeedForwardNeuralNetwork) class
  The model class is the class that keeps track of the Weights and biases and does call the optimizer class and routines for weight updates for training and does a forward call when inputs_ are given and also has options to threshold the outputs.

 - "__init__" is used to intiate the variables that are crucial to the model such as architecture, activation_sequence, loss, Weights and Biases, optimizer variables, and intiation scheme etc. Note that the weights and biases are initiated using the "init_mat()" function defined in the first section of the readme file.

 - "forward_call()" method is calling the "forward_propagation()" method implicitly to find the output of forward propagation through the network and get the outputs and perform centain additional computations such as thresholding or finding the logits etc.

 - "update_params()" this method is called when we have to update the Weights and Biases stores in the model class as a self variable.

 - "train_step()" method is called when we have to perform one step of training. It gets input of the X_batch, y_batch, epoch and does the following
   1) finds the pre-activations and post-activations by calling the "forward_propagation()" method.
   2) passes this to the optimizer.stepper() function which does one step of gradient descent mentioned and returns the weights and biases.
   3) Then update the parameters of the model with "update_params()" method.
  

## Model Training
 The flow of the training of the feed forward neural network is shown below in the given diagram. The train loop is in the bottom-right which call the the methods and classes in the training module.
 
![DL drawio](https://github.com/user-attachments/assets/04209272-1bd1-4199-8001-b8ecee5a9933)

## Model Validating
 The validation of the model can be added in the train.py if required. By default the confusion matrix for the test data, the train and validation data accuracy is logged at every training epoch. Final test accuracy is also logged at wandb. Do feel free to add and log any other metric.

## Add a new optimizer
 Lets say we are to add new optimizer named EVE !!! Now the steps to follow are these, add a function under optimizer named "eve_step()" which when given the gradients does one step of gradient descent. Now we will have to add the parameters and some additional matrices for this add those in the __init__() function in the optimizer class. Also I have an assert in the optimizers do take care to add the new optimizer name in that. Then in the "stepper()" function in the optimizer add the new optimizer by following the syntax of already available optimizer. Voila its done !!!
