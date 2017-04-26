from __future__ import absolute_import

from ..initializers import Initializer

class Embeddings(Layer):
    """
    Embeddings layer.

    This class initializes an embedding layer, which transforms tokens into continuous space representation

    # Arguments
        name: name used to identify the layer object
        vocab_size: size of the input = size of vocabulary
        emb_dim: word embeddings dimension
        initializer: method to initialize the embeddings
    """    
    def __init__(self, name, vocab_size, emb_dim, initializer=Initializer.norm):
        self.in_dim = vocab_size
        self.out_dim = emb_dim
        self.initialize_params(name, vocab_size, emb_dim, initializer)
        
    def initialize_params(self, name, vocab_size, emb_dim, initializer=Initializer.norm):
        self._params = OrderedDict()
        self.Emb = theano.shared(name=name, value=initializer(size=(vocab_size, emb_dim)))
        self._params[self.Emb.name] = self.Emb
    
    def params(self):
        return self._params
    