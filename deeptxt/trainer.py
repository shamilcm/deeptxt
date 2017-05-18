import theano
import numpy as np
import logging


from .utils import logutils
logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model, optimizer, loss=None):
        """
        """
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self.forward, self.backward = optimizer.minimize(model)
        model.build_sampler()

        self.num_updates = 0

        self.f_debug = None
        if hasattr(model, 'debug') and model.debug and None not in model.debug:
            self.f_debug = theano.function(model.inputs(), model.debug, on_unused_input='ignore')

    def loss():
        return loss

    def update(self, minibatch, max_length=None, verbose=False):
        inp = self._model.prepare_train_input(minibatch, max_length)
        if inp is None:
            return None

        if verbose == True and self.f_debug:
            logger.debug('Debug Message:' + str(self.f_debug(*inp)))

        train_loss = self.forward(*inp) #Calculate loss and gradient
        self.backward()  # Update prameters
        self.num_updates += 1
        return train_loss

    def sampling(self, minibatch, num_samples=5):
        samples = self._model.sample(minibatch, num_samples)   # sample is a tuple of (source, output, reference)
        for i in xrange(len(samples)):
            logger.info('Sample %d' %i)
            for label, value in samples[i].iteritems():
                logger.info('\t' + logutils.white(label) + ': ' + value)


    def save_model(self, model_path, validator = None):
        # TODO: saving best_p separately
        # TODO: nematus compatibility, change names?
        # TODO: change the save_file
        params = dict([ ( param_name, param_var.get_value() )
                         for param_name, param_var in self._model.params().iteritems()])
        np.savez(model_path, validation_history= validator.history(), num_updates= self.num_updates, model_hyperparams= self._model.hyperparams(), **params)


    def load_model(self, model_path, validator=None):
        loaded_arrs = np.load(model_path)
        if validator:
            # TODO: setter method
            validator._history = loaded_arrs['validation_history']
        self._model.load_params(loaded_arrs)
        self.num_updates =  loaded_arrs['num_updates']


    def log_train_info(self, epoch, loss, num_batches=None,  time_per_update=None):
        log_msg = 'Epoch#: %d'  % epoch
        if num_batches:
            log_msg += '\tMinibatch#: %d' % num_batches
        log_msg += '\tUpdate#: %d' % self.num_updates
        log_msg += '\tAvg. Loss: ' + "{0:.4f}".format(loss/num_batches)

        if time_per_update:
            log_msg +=  '\tTime/Update: ' + "{0:.4f}".format(time_per_update) + 's'

        logger.info(log_msg)
