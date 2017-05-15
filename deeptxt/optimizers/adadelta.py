from __future__ import absolute_import

import theano
from theano import tensor as T

from .optimizer import Optimizer
from ..nn.initializers import Initializer

class Adadelta(Optimizer):

    def __init__(self, learning_rate=None, decay_factor=0.95, epsilon=1e-6, **kwargs):

        super(Adadelta, self).__init__(**kwargs)
        self.learning_rate = None  # does not use the learning rate parameters
        self.decay_factor = decay_factor  ## rho for computing decaying average
        self.epsilon = epsilon


    def minimize(self, model):
        grads = [theano.shared(name='%s_grad' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        grad_decayavg_sqrsums = [theano.shared(name='%s_grad_sqrsums' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]
        deltaparam_decayavg_sqrsums = [theano.shared(name='%s_delta_sqrsums' % param_name, value=Initializer.zeros(param.get_value().shape))
                                 for param_name, param in model.params().iteritems()]

        # Forward pass - loss and gradient calculation
        grad_exprs = self.get_gradients(model)
        grad_updates = [(grad, grad_expr) for grad, grad_expr in zip(grads, grad_exprs)]
        # E[g^2]_t = rho x E[g^2]_{t-1} + (1-rho) x g^2_t
        grad_decayavg_sqrsum_updates = [(grad_decayavg_sqrsum, self.decay_factor * grad_decayavg_sqrsum + (1. - self.decay_factor) * grad_expr ** 2 )
                              for grad_decayavg_sqrsum, grad_expr in zip(grad_decayavg_sqrsums, grad_exprs)]
        self.forward = theano.function(model.inputs(), model.loss(), updates=grad_updates + grad_decayavg_sqrsum_updates)

        # Backward pass - parameter updates
        deltaparams = [ -T.sqrt(deltaparam_decayavg_sqrsum + self.epsilon) * grad / T.sqrt(grad_decayavg_sqrsum + self.epsilon)
                       for deltaparam_decayavg_sqrsum, grad_decayavg_sqrsum, grad
                       in zip (deltaparam_decayavg_sqrsums, grad_decayavg_sqrsums, grads)]

        deltaparam_decayavg_sqrsum_updates = [(deltaparam_decayavg_sqrsum, self.decay_factor * deltaparam_decayavg_sqrsum + (1. - self.decay_factor) * deltaparam ** 2 )
                              for deltaparam_decayavg_sqrsum, deltaparam in zip(deltaparam_decayavg_sqrsums, deltaparams)]

        param_updates = [(param, param + deltaparam)
                         for param, deltaparam in zip(model.params().values(), deltaparams) ]
        self.backward = theano.function([], [], updates=param_updates + deltaparam_decayavg_sqrsum_updates )

        return self.forward, self.backward
