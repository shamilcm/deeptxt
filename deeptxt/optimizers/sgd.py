from __future__ import absolute_import
import theano
from theano import tensor as T

class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer

    Implementation of the SGD optimization algorithms

     # Arguments
        learning_rate: the learning_rate for the optimizer
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def minimize(self, model):
        
        grad_exprs = T.grad(model.loss().mean(), wrt=model.params().values())
        
        # TODO: better way of setting 0s
        grads = [theano.shared(param.get_value() * 0., name='%s_grad' % param_name)
                          for param_name, param in model.params().iteritems()]
        
        # Create update iterables ([(gradient varible, gradient expression) ...])
        # Update grad variables with grad expressions
        grad_updates = [(grad, grad_expr) for grad, grad_expr in zip(grads, grad_exprs)]
        
        
        #TODO: combine cost and update functions
        self.f_loss = theano.function(model.inputs(), model.loss(), updates=grad_updates)
        
        # Update params after computing the gradients for the cost
        param_updates = [(param, param - self.learning_rate * grad) for param, grad in zip(model.params().values(), grads)]
        self.f_update = theano.function([], [], updates=param_updates)
        
        return (self.f_loss, self.f_update)
        