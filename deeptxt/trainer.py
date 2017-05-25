import theano
import numpy as np
import logging
import time


from .utils import logutils
logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model, optimizer, loss=None, cache_minibatches=False):
        """
        """
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self.forward, self.backward = optimizer.minimize(model)
        model.build_sampler()

        self.num_updates = 0

        self.cache_minibatches = cache_minibatches
        self.input_cache = dict()


        self.f_debug = None
        if hasattr(model, 'debug') and model.debug and None not in model.debug:
            self.f_debug = theano.function(model.inputs(), model.debug, on_unused_input='ignore')

        # for calculating time and speed
        self.num_tokens_seen = 0
        self.num_samples_seen = 0
        self.last_log_updates = 0
        self.last_log_time = time.time()



    def loss():
        return loss

    def update(self, data_reader,  max_length=None, verbose=False, minibatch_index=None):

        if minibatch_index and self.cache_minibatches == True and minibatch_index in self.input_cache:
            self.current_input = self.input_cache[minibatch_index]
        else:
            minibatch = data_reader.next()
            # reached end of epoch, return None as loss
            if minibatch is None:
                return None
            self.current_input = self._model.prepare_train_input(minibatch, max_length)

            if self.cache_minibatches == True:
                self.input_cache[minibatch_index] = self.current_input

        # For caclulating speed
        self.num_tokens_seen += np.sum(self.current_input[-1])
        self.num_samples_seen += self.current_input[-1].shape[1]

        #print "DEBUG, shape:", [i.shape for i in self.current_input]
        if self.current_input is None:
            return None

        if verbose == True and self.f_debug:
            logger.debug('Debug Message:' + str(self.f_debug(*self.current_input)))

        train_loss = self.forward(*self.current_input) #Calculate loss and gradient
        self.backward()  # Update prameters

        self.num_updates += 1
        return train_loss

    def sampling(self,  num_samples=5):
        samples = self._model.sample(self.current_input, num_samples)   # sample is a tuple of (source, output, reference)
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


    def log_train_info(self, epoch, loss, num_batches=None):

        # calculating duration and time per update
        log_duration = time.time() - self.last_log_time
        time_per_update = log_duration / (self.num_updates - self.last_log_updates)

        log_msg = 'Epoch#: %d'  % epoch
        if num_batches:
            log_msg += '\t| Minibatch#: %d' % num_batches
        log_msg += '\t| Update#: %d' % self.num_updates
        log_msg += '\t| Avg. Loss: ' + "{0:.4f}".format(loss/num_batches)

        log_msg += '\t| Time/Update: ' + "{0:.2f}".format(time_per_update) + ' s'
        log_msg +=  '\t| Speed: ' + str(int(self.num_tokens_seen * 1.0 / log_duration)) + ' words/s , ' +  str(int(self.num_samples_seen * 1.0 / log_duration)) + ' samples/s '

        self.num_tokens_seen = 0
        self.num_samples_seen = 0
        logger.info(log_msg)

        # resetting timer and setting last updates
        self.last_log_time = time.time()
        self.last_log_updates = self.num_updates
