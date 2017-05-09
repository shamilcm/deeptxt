import theano
import numpy as np


class Trainer:
    
    def __init__(self, model, optimizer):
        """
        :param cachce_size: number of batches to cache
        """
        self.model = model
        self.forward, self.backward = optimizer.minimize(model)
        #self.f_log_probs = theano.function(model.inputs(), model.loss())

    def update(self, minibatch, max_length=None):
        inp = self.model.prepare_input(minibatch, max_length)
        #print inp[0].shape, inp[2].shape
        if inp is None:
            return None
        train_loss = self.forward(*inp) #Calculate loss and gradient
        #lp = self.f_log_probs(*inp)
        #print lp, lp.shape
        self.backward()  # Update prameters
        return train_loss
    
    def save(self, save_file, update_number, validator = None):
        # TODO: saving best_p separately
        # TODO: nematus compatibility, change names?
        # TODO: change the save_file
        param_values = [param.get_value() for param in self.model.params().values()]
        np.savez(save_file, validation_history= validator.history(), update_number= update_number, *param_values)
        