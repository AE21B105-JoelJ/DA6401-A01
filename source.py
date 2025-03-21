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

def accuracy(y_hot_pred, y_hot_true):
    """
    input : One hot Y_true, Y_pred
    ouput : accuracy
    """

    total = len(y_hot_true)
    y_pred = np.argmax(y_hot_pred, axis=1)
    y_true = np.argmax(y_hot_true, axis = 1)
    accuracy = np.sum(y_pred == y_true)/total
    return accuracy

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

def softmax(input_, safe = True):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with softmax actiavtion
    """
    if safe: # reduces the overflow in exponential
        inputs_safe = input_ - np.max(input_, axis = 0,keepdims=True)
        output_ = np.exp(inputs_safe )/np.sum(np.exp(inputs_safe), axis = 0, keepdims=True)
    else:
        output_ = np.exp(input_)/np.sum(np.exp(input_), axis = 0, keepdims=True)
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

def diff_softmax_jacob(input_, y_true_reshaped, grad_wrt_postact):
    """
    input : numpy.ndarray 
    Returns :
    output : numpy.ndarray with softmax differentiation actiavtion
    """

    pred_ = softmax(input_)
    grads_wrt_preac = np.zeros_like(pred_, dtype=np.longdouble)
    for i in range(input_.shape[1]):
        jacobian_ = np.diag(pred_[:,i]) - np.outer(pred_[:,i], pred_[:,i])
        grads_ = np.matmul(jacobian_, grad_wrt_postact[:,i])
        grads_wrt_preac[:,i] = grads_

    return grads_wrt_preac


def find_loss(y_pred, y_true, loss = "mean_squared_error"):
    if loss == "mean_squared_error":
        output_ = (1/len(y_true))*np.linalg.norm((y_pred - y_true), ord = "fro")
    elif loss == "binary_cross_entropy":
        output_ = (-1/len(y_true))*np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    elif loss == "cross_entropy":
        eps = 1e-8
        output_ = (-1/len(y_true))*np.sum(np.log(y_pred + eps)*y_true)
    return np.squeeze(output_)

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
    input_reshaped = input_.T
    # Forward prop...
    fp_pre_ac = [input_reshaped]
    fp_post_ac = [input_reshaped]
    for i in range(len(Weights)):
        W, b = Weights[i], Biases[i]
        activation = activation_sequence[i]
        # computing pre activation
        pre_ac = np.matmul(W,input_reshaped) + b
        # appending to the pre activation matrix
        fp_pre_ac.append(pre_ac)
        # computing activation
        if activation in ["sigmoid","ReLu","tanh","softmax","identity"]:
            if activation == "sigmoid":
                post_ac = sigmoid(pre_ac)
            elif activation == "ReLu":
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
        input_reshaped = post_ac.copy()
    
    # output rehaped
    output = fp_post_ac[-1].T
    # return output, preactivations and post activations
    return output, fp_pre_ac, fp_post_ac

class Batchloader:
    """
    Input:
    X_data : Feature data <numpy.ndarray>
    y_data : labels data <numpy.ndarray>
    batch_size : batch size needed int
    shuffle : boolean (if shuffle is needed)
    Output:
    batches : zip(X_batch,y_batch) zip dataloader.
    """
    def __init__(self, X, y, batch_size = 32, shuffle = False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self.ind = np.random.permutation(len(X))
        else:
            self.ind = np.arange(len(X))
        # Initialize the header...
        self.initialize()

    def initialize(self):
        self.head = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.head >= len(self.X):
            self.initialize()
            raise StopIteration # to stop the batchloader
        
        # return the batches of x and y
        tail = self.head + self.batch_size
        X_batch, y_batch = self.X[self.ind[self.head:tail]], self.y[self.ind[self.head:tail]]
        self.head = tail
        return X_batch, y_batch
        
class Optimizer:
    """
    Initialize the optimizer which performs both backpropagation by finding gradients and
    update with respect to the optimizer chosen
    """
    def __init__(self, loss = "mean_squared_error", optimizer = "gd", learning_rate = 0.001, weight_decay = 0, momentum = 0, beta_rms = 0, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-6):
        assert loss in ["mean_squared_error", "binary_cross_entropy", "cross_entropy"], "Loss function is not valid"
        assert optimizer in ["gd","sgd","momentum","nag","adagrad","rmsprop","adam","nadam"], "Optimizer is not valid"
        self.loss = loss
        self.optimizer = optimizer
        self.history = None
        self.learning_rate = learning_rate
        self.iter = 0
        self.eps = eps
        self.weight_decay = weight_decay
        # for momentum based tracking
        self.momentum = momentum
        self.update_mom_w = None
        self.update_mom_b = None
        # for nag tracking
        self.update_nag_w = None
        self.update_nag_b = None
        # for rmsprop
        self.beta_rms = beta_rms
        self.rms_v_w = None
        self.rms_v_b = None
        # for adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.adam_m_w = None
        self.adam_m_b = None
        self.adam_v_w = None
        self.adam_v_b = None
        # for Nadam
        self.nadam_m_w = None
        self.nadam_m_b = None
        self.nadam_v_w = None
        self.nadam_v_b = None
    
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
        y_true_reshaped = y_true.T
        # Firstly find the gradient wrt to output layer
        ## Output activation
        output_activation_str = activation_sequence[-1]
        grads_wrt_postact = np.zeros_like(y_true_reshaped)
        output_ = post_ac[-1]
        input_ = pre_ac[-1]
        if self.loss == "cross_entropy":
            pass # because direct computation of grads_wrt_preac is numerically efficient...
        elif self.loss == "binary_cross_entropy":
            grads_wrt_postact[y_true_reshaped == 1] = - 1/output_[y_true_reshaped == 1] / dim
            grads_wrt_postact[y_true_reshaped == 0] =  1/(1 - output_[y_true_reshaped == 0]) / dim
        elif self.loss == "mean_squared_error":
            grads_wrt_postact = -2*(y_true_reshaped - output_) 
        ## Output layer preactivation
        if output_activation_str == "identity":
            grads_wrt_preac = grads_wrt_postact*diff_linear(input_)
        elif output_activation_str == "softmax":
            if self.loss == "cross_entropy":
                e_l = np.zeros_like(output_)
                e_l[y_true_reshaped == 1] = 1
                grads_wrt_preac = - (y_true_reshaped - output_) / batch_size 
            elif self.loss == "mean_squared_error":
                grads_wrt_preac = diff_softmax_jacob(input_, y_true_reshaped,grads_wrt_postact)
        elif output_activation_str == "sigmoid":
            grads_wrt_preac = grads_wrt_postact*diff_sigmoid(input_)
        elif output_activation_str == "tanh":
            grads_wrt_preac = grads_wrt_postact*diff_tanh(input_)
        elif output_activation_str == "ReLu":
            grads_wrt_preac = grads_wrt_postact*diff_relu(input_)

        for layer in range(1,len(Weights)+1):
            input_ = pre_ac[-layer-1]
            output_ = post_ac[-layer-1]
            W = Weights[-layer]
            b = Biases[-layer]
            # Finding gradients with respect to weights and biases (average along batch size)
            grads_W = (1/batch_size)*np.sum(np.einsum("ij,kj->ikj",grads_wrt_preac,output_),axis=2) #changed mean
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
            if activation == "identity":
                grads_wrt_preac = grads_wrt_postact*diff_linear(input_)
            elif activation == "ReLu":
                grads_wrt_preac = grads_wrt_postact*diff_relu(input_)
            elif activation == "sigmoid":
                grads_wrt_preac = grads_wrt_postact*diff_sigmoid(input_)
            elif activation == "tanh":
                grads_wrt_preac = grads_wrt_postact*diff_tanh(input_)
        
        # assert whether the no_of weights and biasses gradient matrices matches the shape
        assert len(grads_wrt_weights) == len(Weights), "The number of matrices gradient and original do not match"
        assert len(grads_wrt_biases) == len(Biases), "The number of matrices gradient and original do not match"

        return grads_wrt_weights[::-1], grads_wrt_biases[::-1]
    
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
            Weights[i] = Weights[i] - self.learning_rate*grads_wrt_weights[i]
            Biases[i] = Biases[i] - self.learning_rate*grads_wrt_biases[i]

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
            self.update_mom_w[i] = self.momentum*self.update_mom_w[i] + self.learning_rate*grads_wrt_weights[i]
            self.update_mom_b[i] = self.momentum*self.update_mom_b[i] + self.learning_rate*grads_wrt_biases[i]
            # step eqn
            Weights[i] = Weights[i] - self.update_mom_w[i] - self.learning_rate*(2*self.weight_decay*Weights[i])
            Biases[i] = Biases[i] - self.update_mom_b[i] - self.learning_rate*(2*self.weight_decay*Biases[i])
        
        # returning the weights and biases
        return Weights, Biases
    
    def nag_step(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        """
        Does one step gradient descent of weights with nag (does in place)
        Input:
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        grads_wrt_weights : list of gradient weight matrices list[<numpy.ndarray>]
        grads_wrt_biases : list of gradient bias matrices list[<numpy.ndarray>]
        Output: 
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        """

        # Initialize the matrices (if not done already)
        if self.update_nag_w is None or self.update_nag_b is None :
            self.update_nag_b, self.update_nag_w = [], []
            for i in range(len(Weights)):
                self.update_nag_w.append(np.zeros_like(Weights[i],dtype=np.longdouble))
                self.update_nag_b.append(np.zeros_like(Biases[i],dtype=np.longdouble))
        
        # calculating the lookahead Weights and Biases
        Weights_lookahead, Biases_lookahead = [], []
        for i in range(len(Weights)):
            Weights_lookahead.append(Weights[i] - self.momentum*self.update_nag_w[i])
            Biases_lookahead.append(Biases[i] - self.momentum*self.update_nag_b[i])
        
        # Calculating the gradients of the lookahead
        grads_wrt_weights, grads_wrt_biases = self.backprop_grads(Weights_lookahead, Biases_lookahead, pre_ac, post_ac, y_true, activation_sequence)

        # calculating the update and updating the weights
        for i in range(len(Weights)):
            self.update_nag_w[i] = self.momentum*self.update_nag_w[i] + self.learning_rate*grads_wrt_weights[i]
            self.update_nag_b[i] = self.momentum*self.update_nag_b[i] + self.learning_rate*grads_wrt_biases[i]
            # update of the weights
            Weights[i] = Weights[i] - self.update_nag_w[i] - self.learning_rate*(2*self.weight_decay*Weights[i])
            Biases[i] = Biases[i] - self.update_nag_b[i] - self.learning_rate*(2*self.weight_decay*Biases[i])
        
        # returning the weights and biases
        return Weights, Biases
    
    def rmsprop_step(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        """
        Does one step gradient descent of weights with rmsprop (does in place)
        Input:
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        grads_wrt_weights : list of gradient weight matrices list[<numpy.ndarray>]
        grads_wrt_biases : list of gradient bias matrices list[<numpy.ndarray>]
        Output: 
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        """

        eps = 1e-6
        # Initialize the matrices (if not done already)
        if self.rms_v_w is None or self.rms_v_b is None :
            self.rms_v_w, self.rms_v_b = [], []
            for i in range(len(Weights)):
                self.rms_v_w.append(np.zeros_like(Weights[i],dtype=np.longdouble))
                self.rms_v_b.append(np.zeros_like(Biases[i],dtype=np.longdouble))

        # Finding the gradients
        grads_wrt_weights, grads_wrt_biases = self.backprop_grads(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)

        # Update equation anfd updating the weights
        for i in range(len(Weights)):
            self.rms_v_w[i] = self.beta_rms*self.rms_v_w[i] + (1-self.beta_rms)*(grads_wrt_weights[i]**2)
            self.rms_v_b[i] = self.beta_rms*self.rms_v_b[i] + (1-self.beta_rms)*(grads_wrt_biases[i]**2)
            # Update eqn
            Weights[i] = Weights[i] - (self.learning_rate/np.sqrt(self.rms_v_w[i] + eps))*grads_wrt_weights[i] - self.learning_rate*(2*self.weight_decay*Weights[i])
            Biases[i] = Biases[i] - (self.learning_rate/np.sqrt(self.rms_v_b[i] + eps))*grads_wrt_biases[i] - self.learning_rate*(2*self.weight_decay*Biases[i])

        # returning the weights and biases
        return Weights, Biases
    
    def adam_step(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        """
        Does one step gradient descent of weights with adam (does in place)
        Input:
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        grads_wrt_weights : list of gradient weight matrices list[<numpy.ndarray>]
        grads_wrt_biases : list of gradient bias matrices list[<numpy.ndarray>]
        Output: 
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        """

        # Initialization of the matrix (if not done already)
        if self.adam_v_w is None or self.adam_v_b is None or self.adam_m_w is None or self.adam_m_b is None:
            self.adam_v_w, self.adam_v_b, self.adam_m_w, self.adam_m_b = [], [], [], []
            for i in range(len(Weights)):
                self.adam_v_w.append(np.zeros_like(Weights[i],dtype=np.longdouble))
                self.adam_v_b.append(np.zeros_like(Biases[i],dtype=np.longdouble))
                self.adam_m_w.append(np.zeros_like(Weights[i],dtype=np.longdouble))
                self.adam_m_b.append(np.zeros_like(Biases[i],dtype=np.longdouble))

        # Finding the gradients
        grads_wrt_weights, grads_wrt_biases = self.backprop_grads(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)

        # Update eqn and updating the weights
        for i in range(len(Weights)):
            # momentum update (with bias correction)
            self.adam_m_w[i] = self.beta_1*self.adam_m_w[i] + (1-self.beta_1)*(grads_wrt_weights[i])
            self.adam_m_b[i] = self.beta_1*self.adam_m_b[i] + (1-self.beta_1)*(grads_wrt_biases[i])
            # adaptive gradient collector update (with bias correction)
            self.adam_v_w[i] = self.beta_2*self.adam_v_w[i] + (1-self.beta_2)*(grads_wrt_weights[i]**2)
            self.adam_v_b[i] = self.beta_2*self.adam_v_b[i] + (1-self.beta_2)*(grads_wrt_biases[i]**2)
            # Bias correction
            self.adam_m_w_hat = self.adam_m_w[i]/(1 - self.beta_1**self.iter)
            self.adam_m_b_hat = self.adam_m_b[i]/(1 - self.beta_1**self.iter)
            self.adam_v_w_hat = self.adam_v_w[i]/(1 - self.beta_2**self.iter)
            self.adam_v_b_hat = self.adam_v_b[i]/(1 - self.beta_2**self.iter)
            # Update eqn
            Weights[i] = Weights[i] - (self.learning_rate*self.adam_m_w_hat)/(np.sqrt(self.adam_v_w_hat) + self.eps) - self.learning_rate*(2*self.weight_decay*Weights[i])
            Biases[i] = Biases[i] - (self.learning_rate*self.adam_m_b_hat)/(np.sqrt(self.adam_v_b_hat) + self.eps) - self.learning_rate*(2*self.weight_decay*Biases[i])

        return Weights, Biases
    
    def nadam_step(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence):
        """
        Does one step gradient descent of weights with Nadam (does in place)
        Input:
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        grads_wrt_weights : list of gradient weight matrices list[<numpy.ndarray>]
        grads_wrt_biases : list of gradient bias matrices list[<numpy.ndarray>]
        Output: 
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        """

        # Initialization of the matrix (if not done already)
        if self.nadam_v_w is None or self.nadam_v_b is None or self.nadam_m_w is None or self.nadam_m_b is None:
            self.nadam_v_w, self.nadam_v_b, self.nadam_m_w, self.nadam_m_b = [], [], [], []
            for i in range(len(Weights)):
                self.nadam_v_w.append(np.zeros_like(Weights[i],dtype=np.longdouble))
                self.nadam_v_b.append(np.zeros_like(Biases[i],dtype=np.longdouble))
                self.nadam_m_w.append(np.zeros_like(Weights[i],dtype=np.longdouble))
                self.nadam_m_b.append(np.zeros_like(Biases[i],dtype=np.longdouble))

        # Finding the gradients
        grads_wrt_weights, grads_wrt_biases = self.backprop_grads(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)

        # Updatind equations
        for i in range(len(Weights)):
             # momentum update (with bias correction)
            self.nadam_m_w[i] = self.beta_1*self.nadam_m_w[i] + (1-self.beta_1)*(grads_wrt_weights[i])
            self.nadam_m_b[i] = self.beta_1*self.nadam_m_b[i] + (1-self.beta_1)*(grads_wrt_biases[i])
            # adaptive gradient collector update (with bias correction)
            self.nadam_v_w[i] = self.beta_2*self.nadam_v_w[i] + (1-self.beta_2)*(grads_wrt_weights[i]**2)
            self.nadam_v_b[i] = self.beta_2*self.nadam_v_b[i] + (1-self.beta_2)*(grads_wrt_biases[i]**2)
            # Bias correction
            self.nadam_m_w_hat = self.nadam_m_w[i]/(1 - self.beta_1**self.iter)
            self.nadam_m_b_hat = self.nadam_m_b[i]/(1 - self.beta_1**self.iter)
            self.nadam_v_w_hat = self.nadam_v_w[i]/(1 - self.beta_2**self.iter)
            self.nadam_v_b_hat = self.nadam_v_b[i]/(1 - self.beta_2**self.iter)
            # Update eqn
            Weights[i] = Weights[i] - (self.learning_rate/(np.sqrt(self.nadam_v_w_hat) + self.eps))*(self.beta_1*self.nadam_m_w_hat + ((1-self.beta_1)/(1 - self.beta_1**self.iter))*grads_wrt_weights[i]) - self.learning_rate*(2*self.weight_decay*Weights[i])
            Biases[i] = Biases[i] - (self.learning_rate/(np.sqrt(self.nadam_v_b_hat) + self.eps))*(self.beta_1*self.nadam_m_b_hat + ((1-self.beta_1)/(1 - self.beta_1**self.iter))*grads_wrt_biases[i]) - self.learning_rate*(2*self.weight_decay*Biases[i])

        return Weights, Biases

    def stepper(self, Weights, Biases, pre_ac, post_ac, y_true, activation_sequence, epoch):
        """
        Does one step update of weights with the optimizer chosen (Helper function called from model)
        Input:
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        grads_wrt_weights : list of gradient weight matrices list[<numpy.ndarray>]
        grads_wrt_biases : list of gradient bias matrices list[<numpy.ndarray>]
        Output: 
        Weights : list of weight matrices list[<numpy.ndarray>]
        Biases : list of bias matrices list[<numpy.ndarray>]
        """

        self.iter += 1
        # Step according to the optimizer
        if self.optimizer == "sgd":
            self.momentum = 0
            Weights, Biases = self.sgd_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        elif self.optimizer == "momentum":
            Weights, Biases = self.sgd_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        elif self.optimizer == "gd":
            Weights, Biases =  self.gd_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        elif self.optimizer == "nag":
            Weights, Biases = self.nag_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        elif self.optimizer == "rmsprop":
            Weights, Biases = self.rmsprop_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        elif self.optimizer == "adam":
            Weights, Biases = self.adam_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        elif self.optimizer == "nadam":
            Weights, Biases = self.nadam_step(Weights, Biases, pre_ac, post_ac, y_true, activation_sequence)
        return Weights, Biases

        
class FeedForwardNeuralNetwork:
    """
    Feed forward neural network class which is used to train the model and store the weights...
    in short (An Orchestrator of the modules)
    """
    def __init__(self, arch : List , activation_sequence : List, optimizer = "adam", learning_rate = 1e-3, eps = 1e-6, weight_decay = 0, loss = "cross_entropy", initialization = "Xavier", momentum = 0,threshold = 0.5, 
                beta_rms = 0.95, beta_1 = 0.9, beta_2 = 0.999):
        self.arch = arch
        self.activation_seqence = activation_sequence
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.learning_rate  = learning_rate
        self.eps = eps
        self.loss = loss
        self.momentum = momentum
        self.initialization = initialization
        self.threshold = threshold
        self.beta_rms = beta_rms
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.Optimizer_class = Optimizer(loss=self.loss,optimizer=self.optimizer,learning_rate=self.learning_rate,weight_decay=self.weight_decay,momentum=self.momentum,
                                        beta_rms=self.beta_rms, beta_1 = self.beta_1, beta_2 = self.beta_2, eps=self.eps)
        # Some assertions to be made
        assert len(self.activation_seqence) == len(self.arch) - 1 , "Number of layers and activation do not match"

        # Initialization of the weights
        self.weights, self.biases = init_mat(Info=self.arch, init_scheme= self.initialization)

    def forward_call(self, inputs_, is_batch_alone = False, is_batch_both = False, with_logits = False, threshold = False):
        """
        Forward call with the inputs
        inputs_ : <numpy.ndarra> or batchloader 
        is_batch_alone : boolean whether the input is a batch of X alone
        is_batch_both : boolean whether the input is a batch of both X,Y
        with_logits : boolean whether not to apply the output activation
        threshold: boolean whether the threshold has to be applied
        Output
        output_final : <numpy.ndarray> Output from the model
        """
        activation_copy = self.activation_seqence.copy()
        if with_logits:
            activation_copy[-1] = "identity"
        outputs_ = None
        if is_batch_alone:
            for X in inputs_:
                out_batch, _, _ = forward_propagation(X, self.weights, self.biases, activation_sequence=activation_copy)
                if outputs_ is None:
                    outputs_ = out_batch.copy()
                else:
                    outputs_ = np.append(outputs_, out_batch, axis=0)

        elif is_batch_both:
            for X, _ in inputs_:
                out_batch, _, _ = forward_propagation(X, self.weights, self.biases, activation_sequence=activation_copy)
                if outputs_ is None:
                    outputs_ = out_batch.copy()
                else:
                    outputs_ = np.append(outputs_, out_batch, axis=0)
        else:
            outputs_, _, _ = forward_propagation(inputs_, self.weights, self.biases, activation_sequence=activation_copy)

        # threshold the outputs
        out_final = outputs_.copy()
        if threshold and self.activation_seqence[-1] == "softmax":
            out_final = np.zeros_like(outputs_, dtype= np.int64)
            out_final[np.arange(len(outputs_)),np.argmax(outputs_,axis=1)] = 1
        elif threshold and self.activation_seqence[-1] == "sigmoid":
            out_final = np.zeros_like(outputs_, dtype= np.int64)
            out_final[outputs_ >= self.threshold] = 1
            out_final[outputs_ < self.threshold] = 0
        return out_final
    
    def update_params(self, Weights, Biases):
        """
        Update the weights and biases of the model
        """
        self.weights = Weights.copy()
        self.biases = Biases.copy()
        return self

    def train_step(self, X_train, y_train, epoch):
        """
        Does one step of training of the model
        """
        # Forward propagation
        _, preac, postac = forward_propagation(X_train, self.weights, self.biases, activation_sequence = self.activation_seqence)
        # Do one optimizer step
        Weights, Biases = self.Optimizer_class.stepper(self.weights, self.biases, preac, postac, y_train, self.activation_seqence, epoch=epoch)
        # Update the weights
        self.update_params(Weights, Biases)
