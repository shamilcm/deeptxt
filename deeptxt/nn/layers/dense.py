
from __future__ import absolute_import
from ..initializers import Initializer
from ..activations import Activation

import theano
from theano import tensor as T
from collections import OrderedDict

from .layer import Layer

class Dense(Layer):
    """
    Dense layer

    An affine transformation with optional non-linear activation function

    # Arguments
        name: name to indetify the layer
        in_dim: input dimension
        num_units: output dimension
        activation: activation function
        weight_initializer: initialization method to initialize weights
        bias_initializer: initialization method to initialize biases
    """
    def __init__(self, name, in_dim, num_units, activation=Activation.tanh, weight_initializer=Initializer.norm, bias_initializer=Initializer.zeros):
        self.name = name
        self.in_dim = in_dim
        self.out_dim = num_units
        self.activation = activation
        self.initialize_params(name, weight_initializer, bias_initializer)
    
    def initialize_params(self, name, weight_initializer, bias_initializer):
        self._params = OrderedDict()
        
        # Initializing W
        if weight_initializer == Initializer.norm:
            param_value = weight_initializer(size=(self.in_dim, self.out_dim), orthogonal=False)
        else:
            param_value = weight_initializer(size=(self.in_dim, self.out_dim))
        self.W = theano.shared(name=self.name + '_W', value=param_value)
        self._params[self.W.name] = self.W

        # Initializing bx
        if bias_initializer:
            self.b = theano.shared(name=self.name + '_b', value=bias_initializer((self.out_dim,)))
            self._params[self.b.name] = self.b
    
    def params(self):
        return self._params
    
    def build(self, x):
        self._input = x
        self._output = self.activation(T.dot(x, self.W) + self.b)
        return self._output
    
    def outputs():
        return self._output