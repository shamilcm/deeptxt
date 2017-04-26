from __future__ import absolute_import
import numpy as np

class Initializer:
    
    dtype ='float32'
    
    @staticmethod
    def ortho(size):
        if len(size) == 2 and size[0] == size[1]:
            W = np.random.normal(size=size)
            U, S, V = np.linalg.svd(W)
            return U.astype(Initializer.dtype)
        else:
            print >> sys.stderr, 'Orthogonal initialization requires nxn size!'

    @staticmethod
    def norm(size, scale=0.01, orthogonal=True):
        ## TODO: if size[1] is None, size[0] = size[1]
        if orthogonal == True and len(size) == 2 and size[0] == size[1]:
            W = Initializer.ortho(size)
        else:
            W = np.random.normal(size=size, scale=scale)
        return W.astype(Initializer.dtype)

    @staticmethod
    def zeros(size):
        return np.zeros(size).astype(Initializer.dtype)
    
    @staticmethod
    def prepare_sharedvars(init_vals_dict):
        sharedvars_dict = OrderedDict()
        for name, value in init_vals_dict.iteritems():
            sharedvars_dict[name] = theano.shared(value=value, name=name)
        return sharedvars_dict