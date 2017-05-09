from __future__ import absolute_import

from ..initializers import Initializer
import theano
from theano import tensor as T
from collections import OrderedDict

from .layer import Layer

class Embeddings(Layer):
    """
    Embeddings layer.

    This class initializes an embedding layer, which transforms tokens into continuous space representation

    # Arguments
        name: name used to identify the layer object
        vocab_size: size of the input = size of vocabulary
        emb_dim: word embeddings dimension
        initializer: method to initialize the embeddings
        allow_bos: allow a 0-vector for the bos token
    """    
    def __init__(self, name, vocab_size, emb_dim, initializer=Initializer.norm, add_bos=False):
        if add_bos == True:
            vocab_size = vocab_size + 1
        self.add_bos = add_bos
        self.in_dim = vocab_size
        self.out_dim = emb_dim
        self.initialize_params(name, vocab_size, emb_dim, initializer)
        
    def initialize_params(self, name, vocab_size, emb_dim, initializer=Initializer.norm):
        self._params = OrderedDict()
        embeddings = initializer(size=(vocab_size, emb_dim))
        if self.add_bos:
            embeddings[-1] = 0.
        self.Emb = theano.shared(name=name, value=embeddings)
        self._params[self.Emb.name] = self.Emb
    
    def params(self):
        return self._params
    