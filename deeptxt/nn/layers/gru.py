from __future__ import absolute_import
from ..initializers import Initializer
from ..activations import Activation

import theano
from theano import tensor as T
import numpy as np
from collections import OrderedDict

from .layer import Layer
from ..tensor import Tensor

class GRU(Layer):
    
    def __init__(self, name, in_dim, num_units, weight_initializer=Initializer.norm, bias_initializer= Initializer.zeros):
        self.name = name
        self.in_dim = in_dim
        self.num_units = num_units
        self.out_dim = num_units
        self.initialize_params(in_dim, num_units, weight_initializer, bias_initializer)
        
    def initialize_params(self, in_dim, num_units, weight_initializer=Initializer.norm, bias_initializer=Initializer.zeros):
        self._params = OrderedDict()
        
        # Initializing W = [Wr, Wu]. dim = in_dim x (2*num_units)
        self.W = theano.shared(name=self.name + '_W', value=np.concatenate([weight_initializer((in_dim, num_units)),
                                                                            weight_initializer((in_dim, num_units))], axis=1)) #concat horizontally
        self._params[self.W.name] =  self.W
        
        # Initializing U = [Ur, Uu]. dim = num_units x (2*num_units)
        self.U = theano.shared(name=self.name + '_U', value=np.concatenate([weight_initializer(size=(num_units, num_units)),
                                                                            weight_initializer(size=(num_units, num_units))], axis=1))
        self._params[self.U.name] = self.U

        # Initializing b = [br, bu].  dim = 1 x (2*num_units)
        self.b = theano.shared(name=self.name + '_b', value=bias_initializer((2*num_units,)))
        self._params[self.b.name] = self.b

        # Initializing Wx. dim = in_dim x num_units
        self.Wx = theano.shared(name=self.name + '_Wx', value=weight_initializer((in_dim, num_units)))
        self._params[self.Wx.name] = self.Wx

        # Initializing Ux. dim = num_units x num_units
        self.Ux = theano.shared(name=self.name + '_Ux', value=weight_initializer(size=(num_units, num_units)))
        self._params[self.Ux.name] = self.Ux
        
        # Initializing bx. dim = 1 x num_units
        self.bx = theano.shared(name=self.name + '_bx', value=bias_initializer((num_units,)))
        self._params[self.bx.name] = self.bx
       
    def params(self):
        return self._params
    
    def step(self, x_W_curr, x_Wx_curr, x_mask_curr,         h_prev, U, Ux      ):
        """
        Function for each recurrent step across timesteps
        :param x_W_curr: the dot product of input (current timestep of all samples) and W. 
                          Used to compute r(reset) and u(update) gates
        :param x_Wx_curr: the dot product of  input (current timestep) and Wx. 
                          Used to compute the hidden state proposal.
        :param x_mask_curr: the input mask of the input at current timestep to be multiplied.
        :param h_prev: the previous hidden state.
        """
        inner_matrix = T.dot(h_prev, U) + x_W_curr                        # h_{t-1}.U + x_t.W + b
        # reset gate
        r = Activation.sigmoid(Tensor.slice(inner_matrix, 0, self.num_units)) 
        # update gate
        u = Activation.sigmoid(Tensor.slice(inner_matrix, 1, self.num_units))
                
        # hidden state proposal
        h_prop = Activation.tanh(r * T.dot(h_prev, Ux)  + x_Wx_curr)  # r*(h_{t-1}.Ux) + x_t.W_x + b_X
        # hidden state output
        h = u * h_prev + (1. - u) * h_prop
        # applying the mask on the hidden state
        h = x_mask_curr[:,None] * h + (1. - x_mask_curr)[:,None] * h_prev    
        
        return h
        
    def build(self, x, x_mask):
        """
        Function to build the GRU using the initialized parameters
        :param x: the input to the layer
        :param x_mask: the input mask
        """
        self._input = x
        self._input_mask = x_mask
       
        x_W = T.dot(x, self.W) + self.b
        x_Wx = T.dot(x, self.Wx) + self.bx
        seqs = [x_W, x_Wx, x_mask]
        
        # dim(x) = #timesteps x #samples x #emb_dim
        num_timesteps = x.shape[0]
        num_samples = x.shape[1]
        
        # For each sample in the batch, initialize a 0-vector with "num_units" dimensions 
        # dim(h_init) = #samples * num_units
        h_init = T.unbroadcast(T.alloc(0., num_samples, self.num_units), 0) # Make it unbraodcastable along 0 axis
        
        # Scan function, calls step for each row of the 'seqs'. Arguments pass to step (sequences, outputs_info, non_sequences)
        out, updates = theano.scan(self.step, 
                                  sequences=seqs,
                                  outputs_info=[h_init],
                                  name=self.name+'_Dynamic',
                                  non_sequences=[self.U,self.Ux],
                                  n_steps = num_timesteps,
                                  strict=True) #Strict ensures all shared variables are passed. @TODO: needed?
        
        self._output = [out]
        return self._output
    
    def outputs():
        return self._output