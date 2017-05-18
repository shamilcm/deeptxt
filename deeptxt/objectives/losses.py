from __future__ import absolute_import

import theano
from theano import tensor as T

import logging
from ..utils import logutils 
logger = logging.getLogger(__name__)

def categorical_crossentropy(predictions, targets, mask = None):
    '''
    Returns the categorical cross entropy loss function expression
    Also same as negative log likelihood

    # Arguments
        predictions: a 2-D matrix with dimension with probabilities for each class
                     for each sample. dimension = #samples x #classes
                     or a 3-D tensor for time-series predictions:
                      #timesteps x #samples x #classes
        targets: an 1-D array with #samples elements indicating hte correct class, if predictions is 2-D
                 a 2-D matrix with dimension #timesteps x #samples, where the index of the target class
                 is within every cell
    '''          
    if (predictions.ndim == 3 and targets.ndim == 2) or (predictions.ndim == 2 and targets.ndim == 1):


        # For ease, making targets 2D and predictions 3D
        if predictions.ndim == 2:
          predictions = predictions[None,:,:]

        if targets.ndim == 1:
          targets = targets[None,:]

        
        targets_flat = targets.flatten()
        targets_flat_idx = T.arange(targets_flat.shape[0]) * predictions.shape[-1] + targets_flat
        loss = -T.log(predictions.flatten()[targets_flat_idx])
        loss = loss.reshape([targets.shape[0], targets.shape[1]])
        

        ### Advanced Indexing, Yields wrong results.
        # TODO: find out why this doesn't work!
        #probs = predictions[   T.arange(targets.shape[0]).reshape((-1,1)), T.arange(targets.shape[1]), targets]
        #loss2 = -T.log(probs)
        #debug = predictions.shape, targets.shape, probs, loss1, loss2

        if mask:
            loss = loss * mask
        loss = loss.sum(0).mean()
        
        return loss
    else:
        logger.error('Dimension mismatch for loss function. Predictions and Targets shoudld be either 2D and 1D or 3D and 2D, respectively.')