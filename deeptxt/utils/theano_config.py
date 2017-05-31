def configure(device='cpu', dtype='float32', verbose=False):
    import numpy as np
    np.random.seed(1234)
    import os
    flags = "device=" + device 
    flags += ',floatX=' + dtype
    flags += ',warn_float64=warn' 
    if verbose == True:
        flags += ',exception_verbosity=high'
    #flags += ',optimizer=fast_compile'
    os.environ["THEANO_FLAGS"]  = flags
    import theano
