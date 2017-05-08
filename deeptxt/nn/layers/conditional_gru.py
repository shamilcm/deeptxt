from __future__ import absolute_import
from ..initializers import Initializer
from ..activations import Activation

import theano
from theano import tensor as T
import numpy as np
from collections import OrderedDict

from .layer import Layer
from ..tensor import Tensor

class ConditionalGRU(GRU, Layer):
    
       def __init__(self, name, in_dim, num_units, context_dim,  weight_initializer=Initializer.norm, bias_initializer= Initializer.zeros):
        GRU.__init__(self, name, in_dim, num_units, weight_initializer, bias_initializer)
        self.context_dim = context_dim # Conditioned on the context of encoder as well
        self.initialize_extra_params(weight_initializer, bias_initializer)
        
    def initialize_extra_params(self, weight_initializer, bias_initializer):
        """
        Initialize additional params for the Conditional GRU that is not required
        for GRUs. 
        """
        # Initializing Wc : for context
        self.V = theano.shared(name=self.name + '_V', value=np.concatenate([weight_initializer((self.context_dim, self.num_units)),
                                                                     weight_initializer((self.context_dim, self.num_units))], axis=1)) #concat horizontally
        self._params[self.V.name] = self.V
            
        # Initializing Wcx : for context
        self.Vx = theano.shared(name=self.name + '_Vx', value=weight_initializer(size=(self.context_dim, self.num_units)))
        self._params[self.Vx.name] = self.Vx
        
    def step(self, x_W_curr, x_Wx_curr, x_mask_curr,          h_prev,          c_V, c_Vx, U, Ux ):
        """
        Function for each recurrent step across timesteps
        The input to step function is of the format, (seqs | prev_state | non-seqs)
        :param x_W_curr: the dot product of input (current timestep of all samples) and W. 
                          Used to compute r(reset) and u(update) gates
        :param x_Wx_curr: the dot product of  input (current timestep) and Wx. 
                          Used to compute the hidden state proposal.
        :param x_mask_curr: the input mask of the input at current timestep to be multiplied.
        :param h_prev: the previous hidden state.
        """
        # h_{t-1}.U + c.V + x_t.W + b
        inner_matrix = T.dot(h_prev, U) + x_W_curr  + c_V                     
        # reset gate
        r = Activation.sigmoid(Tensor.slice(inner_matrix, 0, self.num_units)) 
        # update gate
        u = Activation.sigmoid(Tensor.slice(inner_matrix, 1, self.num_units))
                
        # hidden state proposal
        h_prop = Activation.tanh(r * T.dot(h_prev, Ux)  + c_Vx + x_Wx_curr)  # r*(h_{t-1}.U_x) + c.V_x + x_t.W_x + b_x
        # hidden state output
        h = u * h_prev + (1. - u) * h_prop
        # applying the mask on the hidden state
        h = x_mask_curr[:,None] * h + (1. - x_mask_curr)[:,None] * h_prev    
        
        return h
        
    def build(self, x, x_mask, c, h_init=None):
        """
        Function to build the GRU using the initialized parameters
        :param x: the input to the layer
        :param x_mask: the input mask
        :param c: fixed context to condition upon each step
        """
        self._input = x
        self._input_mask = x_mask
        self._input_context = c
       
        # preparing sequence inputs
        x_W = T.dot(x, self.W) + self.b
        x_Wx = T.dot(x, self.Wx) + self.bx
        seqs = [x_W, x_Wx, x_mask]
        
        #preparing context
        c_V = T.dot(c, self.V)
        c_Vx = T.dot(c, self.Vx)
        
        # dim(x) = #timesteps x #samples x #emb_dim
        num_timesteps = x.shape[0]
        num_samples = x.shape[1]
        
        # For each sample in the batch, initialize a 0-vector with "num_units" dimensions 
        # dim(h_init) = #samples * num_units
        if h_init is None:
            h_init = T.unbroadcast(T.alloc(0., num_samples, self.num_units), 0) # Make it unbraodcastable along 0 axis
        
        # Scan function, calls step for each row of the 'seqs'. Arguments pass to step (sequences, outputs_info, non_sequences)
        out, updates = theano.scan(self.step, 
                                  sequences=seqs,
                                  outputs_info=[h_init],
                                  name=self.name+'_Dynamic',
                                  non_sequences=[c_V, c_Vx, self.U,self.Ux],
                                  n_steps = num_timesteps,
                                  strict=True) #Strict ensures all shared variables are passed. @TODO: needed?
        
        self._output = [out]
        return self._output