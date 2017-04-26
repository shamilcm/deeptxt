class Tensor:
    """
    Tensor class

    Class to handle tensor operations
    """   
    @staticmethod
    def slice(x, index, size):
        """ 
         Get one slice from a tensor (3D or 2D), along its last axis
        :param x: the input tensor
        :param index: the index of the slice
        :param size: the size of the slice
        """
        if x.ndim == 3:
            return _x[:,:, index*size:(index+1)*size]
        return x[:, index*size:(index+1)*size]