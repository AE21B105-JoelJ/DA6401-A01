# Source File (Thought to make the core algorithms) #
from typing import List
import numpy as np

# Initialization #
def init_mat(Info : List[int], init_scheme = "random"):
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
            weight_matrix = np.random.randn(Info[i],Info[i-1])*0.1 # Creating weight matrix for each layer
            bias_matrix = np.random.randn(Info[i],1)*0.1 # Creating bias matrix for each layer

        elif init_scheme == "Xavier":
            input_output = Info[i-1] + Info[i]
            std = np.sqrt(2/input_output)
            weight_matrix = np.random.randn(Info[i], Info[i-1])*std # Creating weight matrix for each layer
            bias_matrix = np.random.randn(Info[i], 1)*std # Creating bias matrix for each layer

            # Converting to more precise type
            weight_matrix = weight_matrix.astype(np.longdouble)
            bias_matrix = bias_matrix.astype(np.longdouble)
        # Append the weight and bias matrix ot original list #
        Weights.append(weight_matrix)
        Biases.append(bias_matrix)
    
    return Weights, Biases

def one_hot_numpy(input_):
    """
    input : categorical numpy array (numbers)
    ouput : One hot encoded array
    """

    num_classes = np.max(input_) + 1
    one_hot_enc = np.zeros(shape = (len(input_), num_classes), dtype= np.int64)
    one_hot_enc[np.arange(len(input_)) , input_] = 1.0
    return one_hot_enc

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
    # smoothing for initial phases ...
    #if abs(np.max(input_)) > 100:
    #   input_ = (input_ - np.mean(input_, axis = 0)) / np.var(input_, axis = 0)
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

def diff_softmax(input_, y_reshaped): # Not completed yet...
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with softmax differentiation actiavtion
    """

    softmax_output = softmax(input_)
    output_ = np.zeros_like(input_)
    output_[y_reshaped == 1] = softmax_output[y_reshaped == 1]
    output_ = output_ - softmax_output*(softmax_output[y_reshaped == 1])
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
    assert "softmax" not in activation_sequence[:len(activation_sequence)-1], "softmax (as of now) cant be applied in intermediate layers !!"

    batch_size, dim = input_.shape[0], input_.shape[1]
    input_reshaped = input_.reshape(dim,batch_size)
    # Forward prop...
    fp_pre_ac = [input_reshaped]
    fp_post_ac = [input_reshaped]
    for i in range(len(Weights)):
        W, b = Weights[i], Biases[i]
        activation = activation_sequence[i]
        # computing pre activation
        pre_ac = np.matmul(W,input_reshaped) + b
        # (DEBUG) print(input_reshaped.min(), input_reshaped.max(), W.min(), W.max(), b.min(), b.max())
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
            fp_post_ac.append(post_ac)
        else:
            raise Exception("The activation function is not valid !!")

        # rechanging the input for the next loop instance
        input_reshaped = post_ac
    
    # output rehaped
    output = fp_post_ac[-1].reshape(batch_size,-1)
    # return output, preactivations and post activations
    return output, fp_pre_ac, fp_post_ac

def batchloader(X_data, y_data, batch_size = 32, shuffle = True):
    """
    Input:
    X_data : Feature data <numpy.ndarray>
    y_data : labels data <numpy.ndarray>
    batch_size : batch size needed int
    shuffle : boolean (if shuffle is needed)
    Output:
    batches : zip(X_batch,y_batch) zip dataloader.
    """

    batches_x = []
    batches_y = []
    length_ = len(X_data)
    # Creating the indexed for batching
    if shuffle:
        ind = np.random.permutation(length_)
    else:
        ind = np.arange(length_)
    # num of batches 
    num_batches = (length_ // batch_size) + 1 if length_%batch_size !=0 else length_//batch_size
    for i in range(num_batches):
        if i == num_batches - 1:
            batches_x.append(X_data[i*batch_size:])
            batches_y.append(y_data[i*batch_size:])
            break
        batches_x.append(X_data[i*batch_size:(i+1)*batch_size])
        batches_y.append(y_data[i*batch_size:(i+1)*batch_size])
    # returning a zip of the batch
    return (batches_x, batches_y)

class Batchloader:
    def __init__(self, X, y, batch_size = 32, shuffle = False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self.ind = np.random.permutation(len(X))
        else:
            self.ind = np.arange(len(X))

        self.initialize()

    def initialize(self):
        self.head = 0

    def __iter__(self):
        return self
    
    
        

class Optimizer:
    def __init__(self, loss = "mean_squared_error", optimizer = "gd", learning_rate = 0.001, momentum = 0):
        assert loss in ["mean_squared_error", "binary_cross_entropy", "cross_entropy"], "Loss function is not valid"
        assert optimizer in ["gd","sgd","mom","nag","adagrad","rmsprop","adam"], "Optimizer is not valid"
        self.loss = loss
        self.optimizer = optimizer
        self.history = None
        self.learning_rate = learning_rate
        # for momentum based tracking
        self.momentum = momentum
        self.update_mom_w = None
        self.update_mom_b = None
    
    def backprop_grads(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        """
        Input :
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        pre_ac : pre-activations of all layers list[<numpy.ndarray>]
        post_ac : post-activations of all layers list[<numpy.ndarray>]
        y_true : true labels <numpy.ndarray>
        activation_sequence : activation of each layer list[str]
        Output:
        grads_wrt_w : list of gradient of W matrices list[<numpy.ndarray>]
        grads_wrt_b : list of gradient of b matrices list[<numpy.ndarray>]
        """

        grads_wrt_postact = None
        grads_wrt_preac = None
        grads_wrt_weights = []
        grads_wrt_biases = []
        # Reshape y_true wrt our convention (dim, batch_size)
        batch_size, dim = y_true.shape[0], y_true.shape[1]
        y_true_reshaped = y_true.reshape(dim,batch_size)
        # Firstly find the gradient wrt to output layer
        ## Output activation
        output_activation_str = activation_sequence[-1]
        grads_wrt_postact = np.zeros_like(y_true_reshaped)
        output_ = post_ac[-1]
        input_ = pre_ac[-1]
        if self.loss == "cross_entropy":
            grads_wrt_postact[y_true_reshaped == 1] = - 1/output_[y_true_reshaped == 1]
        elif self.loss == "binary_cross_entropy":
            grads_wrt_postact[y_true_reshaped == 1] = - 1/output_[y_true_reshaped == 1]
            grads_wrt_postact[y_true_reshaped == 0] =  1/(1 - output_[y_true_reshaped == 0])
        elif self.loss == "mean_squared_error":
            grads_wrt_postact = -2*(y_true_reshaped - output_)*output_
        ## Output layer preactivation
        if output_activation_str == "linear":
            grads_wrt_preac = grads_wrt_postact*diff_linear(input_)
        elif output_activation_str == "softmax":
            grads_wrt_preac = (-1/output_[y_true_reshaped == 1])*diff_softmax(input_,y_true_reshaped)
        elif output_activation_str == "sigmoid":
            grads_wrt_preac = grads_wrt_postact*diff_sigmoid(input_)
        elif output_activation_str == "tanh":
            grads_wrt_preac = grads_wrt_postact*diff_tanh(input_)
        elif output_activation_str == "relu":
            grads_wrt_preac = grads_wrt_postact*diff_relu(input_)

        for layer in range(1,len(Weights)+1):
            input_ = pre_ac[-layer-1]
            output_ = post_ac[-layer-1]
            W = Weights[-layer]
            b = Biases[-layer]
            # Finding gradients with respect to weights and biases (average along batch size)
            # (- checker ) print(f" Layer : {-layer}  \n W : {W.shape} \n B : {b.shape} \n preac : {grads_wrt_preac.shape}")
            grads_W = (1/batch_size)*np.sum(np.einsum("ij,kj->ikj",grads_wrt_preac,output_),axis=2)
            grads_b = (1/batch_size)*np.sum(grads_wrt_preac, axis = 1,keepdims=True)
            # check the shapes of the gradient matches with the matrix size
            assert grads_W.shape == Weights[-layer].shape, f"Shape of grad_W and W at layer : {-layer} does not match"
            assert grads_b.shape == Biases[-layer].shape, f"Shape of grad_b and B at layer : {-layer} does not match"
            # After the gradient shapes match..
            grads_wrt_weights.append(grads_W)
            grads_wrt_biases.append(grads_b)
            if layer == len(Weights):
                break
            activation = activation_sequence[-layer-1]
            # Finding gradients with respect to preactivation and post
            grads_wrt_postact = np.matmul(W.T,grads_wrt_preac)
            if activation == "linear":
                grads_wrt_preac = grads_wrt_postact*diff_linear(input_)
            elif activation == "relu":
                grads_wrt_preac = grads_wrt_postact*diff_relu(input_)
            elif activation == "sigmoid":
                grads_wrt_preac = grads_wrt_postact*diff_sigmoid(input_)
            elif activation == "tanh":
                grads_wrt_preac = grads_wrt_postact*diff_tanh(input_)
        
        # assert whether the no_of weights and biasses gradient matrices matches the shape
        assert len(grads_wrt_weights) == len(Weights), "The number of matrices gradient and original do not match"
        assert len(grads_wrt_biases) == len(Biases), "The number of matrices gradient and original do not match"

        return grads_wrt_weights, grads_wrt_biases
    
    def gd_step(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        """
        Does one step gradient descent of weights (does in place)
        Input:
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        grads_wrt_weights : list of gradient weight matrices list[<numpy.ndarray>]
        grads_wrt_biases : list of gradient bias matrices list[<numpy.ndarray>]
        Output: 
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        """
        grads_wrt_weights, grads_wrt_biases = self.backprop_grads(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        for i in range(len(Weights)):
            Weights[i] = Weights[i] - self.learning_rate*grads_wrt_weights[-i-1]
            Biases[i] = Biases[i] - self.learning_rate*grads_wrt_biases[-i-1]

        return Weights, Biases
    
    def sgd_step(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        """
        Does one step gradient descent of weights with momentum (does in place)
        Input:
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        grads_wrt_weights : list of gradient weight matrices list[<numpy.ndarray>]
        grads_wrt_biases : list of gradient bias matrices list[<numpy.ndarray>]
        Output: 
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        """

        grads_wrt_weights, grads_wrt_biases = self.backprop_grads(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)

        # Initialize the matrices (if not done already)
        if self.update_mom_b is None or self.update_mom_b is None :
            self.update_mom_b, self.update_mom_w = [], []
            for i in range(len(Weights)):
                self.update_mom_w.append(np.zeros_like(Weights[i],dtype=np.longdouble))
                self.update_mom_b.append(np.zeros_like(Biases[i],dtype=np.longdouble))

        # update with momentum
        for i in range(len(Weights)):
            # update eqn
            self.update_mom_w[i] = self.momentum*self.update_mom_w + self.learning_rate*grads_wrt_weights[-i-1]
            self.update_mom_b[i] = self.momentum*self.update_mom_b + self.learning_rate*grads_wrt_biases[-i-1]
            # step eqn
            Weights[i] = Weights[i] - self.update_mom_w[i]
            Biases[i] = Biases[i] - self.update_mom_b[i]
        
        # returning the weights and biases
        return Weights, Biases
    
    def stepper(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        # Step accordin to the optimizer
        if self.optimizer == "sgd":
            return self.sgd_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        elif self.optimizer == "gd":
            return self.gd_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
    
class FeedForwardNeuralNetwork:
    def __init__(self, arch : List , activation_sequence : List, optimizer, learning_rate, loss, initialization, momentum = 0,):
        self.arch = arch
        self.activation_seqence = activation_sequence
        self.optimizer = optimizer
        self.learning_rate  = learning_rate
        self.loss = loss
        self.momentum = momentum
        self.initialization = initialization

        self.Optimizer_class = Optimizer(loss=self.loss,optimizer=self.optimizer,learning_rate=self.learning_rate,momentum=self.momentum)
        # Some assertions to be made
        assert len(self.activation_seqence) == len(self.arch) - 1 , "Number of layers and activation do not match"

        # Initialization of the weights
        self.weights, self.biases = init_mat(Info=self.arch, init_scheme= self.initialization)

    def forward_call(self, inputs_, is_batch_alone = False, is_batch_both = False):
        outputs_list = []
        if is_batch_alone:
            for X in inputs_:
                out_batch, _, _ = forward_propagation(X, self.weights, self.biases, activation_sequence=self.activation_seqence)
                outputs_list.append(out_batch)
            outputs_ = outputs_list[0]
            for i in range(1,len(outputs_list)):
                outputs_ = np.append(outputs_, outputs_list[i], axis = 0)

        elif is_batch_both:
            for X in inputs_[0]:
                out_batch, _, _ = forward_propagation(X, self.weights, self.biases, activation_sequence=self.activation_seqence)
                outputs_list.append(out_batch)
            outputs_ = outputs_list[0]
            for i in range(1,len(outputs_list)):
                outputs_ = np.append(outputs_, outputs_list[i], axis = 0)
        else:
            outputs_, _, _ = forward_propagation(inputs_, self.weights, self.biases, activation_sequence=self.activation_seqence)

        return outputs_
    
    def update_params(self, Weights, Biases):
        self.weights = Weights
        self.biases = Biases

    def train_step(self, X_train, y_train):
        # Forward propagation
        _, preac, postac = forward_propagation(X_train, self.weights, self.biases, activation_sequence = self.activation_seqence)
        # Do one optimizer step
        Weights, Biases = self.Optimizer_class.stepper(self.weights, self.biases, preac, postac, y_train, self.activation_seqence)
        # Update the weights
        self.update_params(Weights, Biases)