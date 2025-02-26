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
        if activation in ["sigmoid","relu","tanh","linear"]:
            if activation == "sigmoid":
                post_ac = sigmoid(pre_ac)
            elif activation == "relu":
                post_ac = relu(pre_ac)
            elif activation == "tanh":
                post_ac = tanh(pre_ac)
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
