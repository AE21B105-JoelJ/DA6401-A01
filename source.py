# Source File (Thought to make the core algorithms) #
from typing import List
import numpy as np

def init_zero_mat(Info : List):
    """
    This functions gets input of the sequence of choises for the hidden layer and neurons per layer and output zero initialized matrices.
    Info : List (Example : [20, 50, 50, 10] Tells that input : 20, first hidden layer : 50, second hidden layer : 50 and output layer : 10) 
    Returns :
    List[numpy.ndarray], List[numpy.ndarray] : 2 List of numpy arrays oe for weights and one for biases.
    """

    