from __future__ import absolute_import

import theano
from theano import tensor as T

from .optimizer import Optimizer
from ..initializers import Initializer

class Adam(Optimizer):
    def __init__(self, learning_rate, decay_rate_mean=0.9, decay_rate_variance=0.999, epsilon=1e-8 ):
        self.beta1 = decay_rate_mean
        self.beta2 = decay_rate_variance
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def minimize(self, model):
        grads = [theano.shared(name='%s_grad' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        mean_decayavgs = [theano.shared(name='%s_mean' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        variance_decayavgs = [theano.shared(name='%s_variance' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]    
        num_updates = theano.shared(name='num_updates', value=Initializer.scalar(0))
        
        # Forward pass - loss and gradient calculation
        grad_exprs = T.grad(model.loss().mean(), wrt=model.params().values())        
        grad_updates = [(grad, grad_expr) for grad, grad_expr in zip(grads, grad_exprs)]

        mean_decayavg_updates = [(mean_decayavg, self.beta1 * mean_decayavg + (1. - self.beta1) * grad_expr)
                                  for mean_decayavg, grad_expr in zip(mean_decayavgs, grad_exprs)]
        variance_decayavg_updates = [(variance_decayavg, self.beta2 * variance_decayavg + (1. - self.beta2) * grad_expr**2)
                                  for variance_decayavg, grad_expr in zip(variance_decayavgs, grad_exprs)]   
        
        self.forward = theano.function(model.inputs(), model.loss(), updates=grad_updates + mean_decayavg_updates + variance_decayavg_updates + [(num_updates, num_updates + 1)])
        
        # Backward pass - parameter updates
        bias_correction = T.sqrt(1. - self.beta2**num_updates) / (1. - self.beta1**num_updates)
        param_updates = [(param, param - self.learning_rate * bias_correction * mean_decayavg / (T.sqrt(variance_decayavg) + self.epsilon) )  
                         for param, grad, mean_decayavg, variance_decayavg in zip(model.params().values(), grads, mean_decayavgs, variance_decayavgs) ]
        self.backward = theano.function([], [], updates=param_updates)   
        
        return self.forward, self.backward