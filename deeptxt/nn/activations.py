class Activation:
    """
    Activation class.

    class for activation functions
    """
    @staticmethod
    def tanh(x):
        return T.tanh(x)
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def sigmoid(x):
        return T.nnet.sigmoid(x)