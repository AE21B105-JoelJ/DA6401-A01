# Source File (Thought to make the core algorithms) #
from typing import List
import numpy as np

# Initialization #
def init_zero_mat(Info : List[int], init_scheme = "random"):
    """
    This functions gets input of the sequence of choises for the hidden layer and neurons per layer and output zero initialized matrices.
    Info : List (Example : [20, 50, 50, 10] Tells that input : 20, first hidden layer : 50, second hidden layer : 50 and output layer : 10) 
    Returns :
    List[numpy.ndarray], List[numpy.ndarray] : 2 List of numpy arrays oe for weights and one for biases.
    """

    init_scheme_all = ["Xavier", "random"]
    if init_scheme not in init_scheme_all:
        raise Exception("The Initializations scheme is either not valid or not appropriate !!!")
    
    Weights, Biases = [], []
    input = Info[0] # Input 
    for i in range(1,len(Info)):
        if init_scheme == "random":
            # Here we consider uniformly random from [-1,1]
            weight_matrix = np.random.uniform(low = -1.0, high = 1.0, size=(Info[i],Info[i-1])) # Creating weight matrix for each layer
            bias_matrix = np.random.uniform(low = -1.0, high = 1.0, size=(Info[i],1)) # Creating bias matrix for each layer

        elif init_scheme == "Xavier":
            input_output = Info[i-1] + Info[i]
            std = np.sqrt(2/input_output)
            weight_matrix = np.random.randn(Info[i], Info[i-1])*std # Creating weight matrix for each layer
            bias_matrix = np.random.randn(Info[i], 1)*std # Creating bias matrix for each layer

        # Append the weight and bias matrix ot original list #
        Weights.append(weight_matrix)
        Biases.append(bias_matrix)
    
    return Weights, Biases

# Activation functions #
def relu(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with relu applied
    """

    zeros_ = np.zeros_like(input_)
    output_ = np.maximum(input_, zeros_)
    return output_

def sigmoid(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with sigmoid applied
    """

    output_ = 1 / (1 + np.exp(-input_))
    return output_

def tanh(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with tanh applied
    """

    output_ = (2 / (1 + np.exp(-2*input_))) - 1
    return output_

def linear(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with nothing applied
    """

    output_ = input_
    return output_

def softmax(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with softmax actiavtion
    """

    output_ = np.exp(input_)/np.sum(np.exp(input_), axis = 0)
    return output_

# Activation functions Derivatives #
def diff_relu(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with differentiated relu applied
    """

    output_ = np.zeros_like(input_)
    output_[input_ > 0] = 1
    return output_

def diff_sigmoid(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with differentiated sigmoid applied
    """

    output_ = sigmoid(input_) * (1 - sigmoid(input_))
    return output_

def diff_tanh(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with differentiated tanh applied
    """

    output_ = 1 - tanh(input_)**2
    return output_

def diff_linear(input_):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with differentiated linear applied
    """

    output_ = np.ones_like(input_)
    return output_

def diff_softmax(input_, ): # Not completed yet...
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with softmax differentiation actiavtion
    """

    softmax_output = softmax(input_)
    output_ = np.exp(input_)/np.sum(np.exp(input_))
    return output_

# Forward propagation #
def forward_propagation(input_, Weights, Biases, activation_sequence : List):
    """
    input : numpy.ndarray - input matrix
    Weights : List of weights in each layer
    Biases : List of biases in each layer
    activation_sequence = List of activation at the end of each layer
    Returns : 
    outs_ :  list of matrices which gives the output, post and pre activations of all layers
    """

    # Some assertions to be made 
    assert input_.shape[1] == Weights[0].shape[1], "The input dimentions does not match !!"
    assert len(Weights) == len(activation_sequence), "The activation sequence does not match with hidden layer !!"
    assert "softmax" in activation_sequence[:len(activation_sequence)-1], "softmax (as of now) cant be applied in intermediate layers !!"

    batch_size, dim = input_.shape[0], input_.shape[1]
    input_reshaped = input_.reshape(dim,batch_size)
    # Forward prop...
    fp_pre_ac = []
    fp_post_ac = []
    for i in range(len(Weights)):
        W, b = Weights[i], Biases[i]
        activation = activation_sequence[i]
        # computing pre activation
        pre_ac = np.matmul(W,input_reshaped) + b
        # appending to the pre activation matrix
        fp_pre_ac.append(pre_ac)
        # computing activation
        if activation in ["sigmoid","relu","tanh","softmax","linear"]:
            if activation == "sigmoid":
                post_ac = sigmoid(pre_ac)
            elif activation == "relu":
                post_ac = relu(pre_ac)
            elif activation == "tanh":
                post_ac = tanh(pre_ac)
            elif activation == "softmax":
                post_ac = softmax(pre_ac)
            else:
                post_ac = linear(pre_ac)
            # appending to the post activation matrix
            fp_post_ac.append(pre_ac)
        else:
            raise Exception("The activation function is not valid !!")

        # rechanging the input for the next loop instance
        input_reshaped = post_ac
    
    # output rehaped
    output = fp_post_ac[-1].reshape(batch_size,-1)
    # return output, preactivations and post activations
    return output, fp_pre_ac, fp_post_ac

class Optimizer:
    def __init__(self, loss = "mean_squared_error", optimizer = "gd"):
        assert loss in ["mean_squared_error", "binary_cross_entropy", "cross_entropy"], "Loss function is not valid"
        assert optimizer in ["gd","sgd","mom","nag","adagrad","rmsprop","adam"]
        self.loss = loss
        self.optimizer = optimizer
        self.history = None
    
    def backprop_grads(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        grads_wrt_postact = None
        grads_wrt_preac = None
        grads_wrt_weights = []
        grads_wrt_biases = []
        # Reshape y_true wrt our convention (dim, batch_size)
        batch_size, dim = y_true.shape[0], y_true.shape[1]
        y_true_reshaped = y_true_reshaped.reshape(dim,batch_size)
        # Firstly find the gradient wrt to output layer
        ## Output activation
        output_activation_str = activation_sequence[-1]
        grads_wrt_postact = np.zeros_like(y_true_reshaped)
        output_ = post_ac[-1]
        if self.loss == "cross_entropy":
            grads_wrt_postact[y_true_reshaped == 1] = - 1/output_[y_true_reshaped == 1]
        elif self.loss == "binary_cross_entropy":
            grads_wrt_postact[y_true_reshaped == 1] = - 1/output_[y_true_reshaped == 1]
            grads_wrt_postact[y_true_reshaped == 0] =  1/(1 - output_[y_true_reshaped == 0])
        elif self.loss == "squared_error_loss":
            grads_wrt_postact = -2*(y_true_reshaped - output_)*output_
        ## Output layer preactivation
        if output_activation_str == "linear":
            grads_wrt_preac = grads_wrt_postact*diff_linear(pre_ac[-1])
        elif output_activation_str == "sigmoid":
            pass