from __future__ import absolute_import

import theano
from theano import tensor as T

from .optimizer import Optimizer
from ..initializers import Initializer

class RMSProp(Optimizer):
    
    def __init__(self, learning_rate, decay_factor=0.90, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor  ## rho for computing decaying average
        self.epsilon = epsilon
        
    def minimize(self, model):
        grads = [theano.shared(name='%s_grad' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        grad_decayavg_sqrsums = [theano.shared(name='%s_grad_sqrsums' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        
        ## Forward pass - loss and gradient calculation
        # finding gradient
        grad_exprs = T.grad(model.loss().mean(), wrt=model.params().values())        
        # creating update list for gradients
        grad_updates = [(grad, grad_expr) for grad, grad_expr in zip(grads, grad_exprs)]
        # E[g^2]_t = rho x E[g^2]_{t-1} + (1-rho) x g^2_t
        grad_decayavg_sqrsum_updates = [(grad_decayavg_sqrsum, self.decay_factor * grad_decayavg_sqrsum + (1. - self.decay_factor) * grad_expr ** 2 )
                              for grad_decayavg_sqrsum, grad_expr in zip(grad_decayavg_sqrsums, grad_exprs)]        
        
        self.forward = theano.function(model.inputs(), model.loss(), updates=grad_updates + grad_decayavg_sqrsum_updates)

        # Backward pass - parameter updates
        param_updates = [(param, param - self.learning_rate * grad / T.sqrt(grad_decayavg_sqrsum + self.epsilon)) 
                         for param, grad, grad_decayavg_sqrsum in zip(model.params().values(), grads, grad_decayavg_sqrsums) ]
        self.backward = theano.function([], [], updates=param_updates)   
        
        return self.forward, self.backward