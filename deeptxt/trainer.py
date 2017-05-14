import theano
import numpy as np
import logging

from .utils import logutils
logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model, optimizer):
        """
        :param cachce_size: number of batches to cache
        """
        self.model = model
        self.forward, self.backward = optimizer.minimize(model)
        model.build_sampler()
        #self.f_debug = theano.function(model.inputs(), model.debug)


    def update(self, minibatch, max_length=None):
        inp = self.model.prepare_input(minibatch, max_length)
        #print "DEBUG:", self.f_debug(*inp)

        #print inp[0].shape, inp[2].shape
        if inp is None:
            return None
        train_loss = self.forward(*inp) #Calculate loss and gradient
        #lp = self.f_log_probs(*inp)
        #print lp, lp.shape
        self.backward()  # Update prameters
        return train_loss

    def sampling(self, minibatch, num_samples=5):
        samples = self.model.sample(minibatch, num_samples)   # sample is a tuple of (source, output, reference)
        for i in xrange(len(samples)):
            logger.info('Sample %d' %i)
            for label, value in samples[i].iteritems():
                logger.info('\t' + logutils.white(label) + ': ' + value)




    def save(self, save_file, update_number, validator = None):
        # TODO: saving best_p separately
        # TODO: nematus compatibility, change names?
        # TODO: change the save_file
        param_values = [param.get_value() for param in self.model.params().values()]
        np.savez(save_file, validation_history= validator.history(), update_number= update_number, *param_values)



    def log_train_info(self, epoch, loss, num_batches=None, updates=None,  time_per_update=None):
        log_msg = 'Epoch: %d'  % epoch
        if num_batches:
            log_msg += '\tBatches: %d' % num_batches
        if updates:
            log_msg += '\tUpdates: %d' % updates

        log_msg += '\tAvg.Loss/Batch: ' + "{0:.4f}".format(loss/num_batches)

        if time_per_update:
            log_msg +=  '\tDuration/Update: %f' % time_per_update

        logger.info(log_msg)
