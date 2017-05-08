from __future__ import absolute_import

import theano
from theano import tensor as T

from .optimizer import Optimizer
from ..initializers import Initializer

class Adagrad(Optimizer):
    
    def __init__(self, learning_rate, epsilon=1e-4):
        # TODO: correct default epsilon?
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def minimize(self, model):
        
        # initializing grad variables
        grads = [theano.shared(name='%s_grad' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        grad_sqrsums = [theano.shared(name='%s_grad' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        # building gradient computation
        grad_exprs = T.grad(model.loss().mean(), wrt=model.params().values())
        grad_updates = [(grad, grad_expr) for grad, grad_expr in zip(grads, grad_exprs)]
        
        grad_sqrsum_updates = [(grad_sqrsum, grad_sqrsum + grad_expr ** 2 )
                              for grad_sqrsum, grad_expr in zip(grad_sqrsums, grad_exprs)]
        self.forward = theano.function(model.inputs(), model.loss(), updates=grad_updates + grad_sqrsum_updates)
        
        #Backward pass
        param_updates = [(param, param - self.learning_rate * grad / T.sqrt(grad_sqrsum + self.epsilon)) 
                         for param, grad, grad_sqrsum in zip(model.params().values(), grads, grad_sqrsums) ]
        self.backward = theano.function([], [], updates=param_updates)
        
        return (self.forward, self.backward)
        