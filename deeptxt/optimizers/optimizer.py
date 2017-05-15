import abc, six

import theano
from theano import tensor as T

from .ops import clip_ops

@six.add_metaclass(abc.ABCMeta)
class Optimizer:
    
    def __init__(self, **kwargs):
        self.clip_norm = None
        self.clip_value = None

        if 'clip_norm' in kwargs:
            self.clip_norm = kwargs['clip_norm']
        if 'clip_value' in kwargs:
            self.clip_value = kwargs['clip_value']

    @abc.abstractmethod
    def minimize(self, model):
        pass

    def get_gradients(self, model):
        gradients = T.grad(model.loss().mean(), wrt=model.params().values())
        if self.clip_norm:
            gradients = clip_ops.clip_by_global_norm(gradients, threshold=self.clip_norm)
        if self.clip_value:
            gradients = clip_ops.clip_by_value(gradients, min_threshold=-self.clip_value, max_threshold=self.clip_value)
        return gradients
    
