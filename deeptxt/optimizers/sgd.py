from __future__ import absolute_import

import theano
from theano import tensor as T

from .optimizer import Optimizer
from ..initializers import Initializer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer

    Implementation of the SGD optimization algorithm

     # Arguments
        learning_rate: the learning_rate for the optimizer
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def minimize(self, model):
        
        grad_exprs = T.grad(model.loss().mean(), wrt=model.params().values())
        grads = [theano.shared(name='%s_grad' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]       
        
        # Create update iterables ([(gradient varible, gradient expression) ...])
        # Update grad variables with grad expressions
        grad_updates = [(grad, grad_expr) for grad, grad_expr in zip(grads, grad_exprs)]
        self.forward = theano.function(model.inputs(), model.loss(), updates=grad_updates)
        
        # Update params after computing the gradients for the cost
        param_updates = [(param, param - self.learning_rate * grad) for param, grad in zip(model.params().values(), grads)]
        self.backward = theano.function([], [], updates=param_updates)
        
        return (self.forward, self.backward)
        