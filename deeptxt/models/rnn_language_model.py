
from __future__ import absolute_import

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from collections import OrderedDict

from .model import Model
from ..nn.layers.embeddings import Embeddings
from ..nn.layers.gru import GRU
from ..nn.layers.dense import Dense
from ..nn.activations import Activation

class RNNLanguageModel(Model):
    
    def __init__(self, hyperparams, vocab):
        self._hyperparams = hyperparams
        self.vocab = vocab
        # Preparing and Initializing Network Weights & Biases
        self.setup()

    def hyperparams(self):
        return self._hyperparams
        
    # TODO: Loading and storing params
    def setup(self):
        '''
            Setup the shared variables and model components
        '''
        self._params = OrderedDict()
        self.embeddings = Embeddings('Emb', self.vocab.vocab_size, self.hyperparams().emb_dim, add_bos=True)
        self._params.update(self.embeddings.params())
        
        if self.hyperparams().rnn_cell == 'gru':
            self.rnn_layer = GRU(name='gru0', in_dim=self.hyperparams().emb_dim, num_units=self.hyperparams().num_units)
            self._params.update(self.rnn_layer.params())
        elif self.hyperparams().rnn_cell == 'lstm':
            raise NotImplementedError
        else:
            logger.error("Invalid RNN Cell Type:" + self.hyperparams().rnn_cell)
        
        # TODO : Cleanup this part of code!
        self.ff_logit_rnn = Dense(name='ff_logit_rnn', in_dim=self.hyperparams().num_units, num_units=self.hyperparams().emb_dim, activation=Activation.linear)
        self._params.update(self.ff_logit_rnn.params())
        
        self.ff_logit_prev = Dense(name='ff_logit_prev', in_dim=self.hyperparams().emb_dim, num_units=self.hyperparams().emb_dim, activation=Activation.linear)
        self._params.update(self.ff_logit_prev.params())

        self.ff_logit = Dense(name='ff_logit', in_dim=self.hyperparams().emb_dim, num_units=self.vocab.vocab_size, activation=Activation.linear)
        self._params.update(self.ff_logit.params())
        # DEBUG
        #for k, v in self._params.iteritems():
        #    print k, v.get_value(), v.get_value().shape
        
    def build(self):
        self.trng = RandomStreams(1234)
        
        # dim(x) = (time_steps, num_samples)
        self.x = T.matrix('x', dtype='int64')
        # dim(x) = (time_steps, num_samples)
        self.x_mask = T.matrix('x_mask', dtype='float32')
        
        # input
        # TODO: remove embedding shifting?
        emb = self.embeddings.Emb[self.x.flatten()]
        emb = emb.reshape([self.x.shape[0], self.x.shape[1], self.hyperparams().emb_dim])
        
        # Building the RNN layer
        proj_h = self.rnn_layer.build(emb,  self.x_mask)[0]  # Only one output, hidden state
        # dim(proj_h) = #timesteps x #samples x #num_units
        
        # Computing word probabilities
        logit_rnn = self.ff_logit_rnn.build(proj_h)  # dim(logit_rnn) = #timesteps x #samples x #emb_dim
        logit_prev = self.ff_logit_prev.build(emb)   # dim(logit_prev) = #timesteps x #samples x #emb_dim
        logit = self.ff_logit.build(Activation.tanh(logit_rnn + logit_prev)) # dim(logit) = #timesteps x #samples x #vocab_size
        
        # reshaping logit as (#timesteps*#samples) x vocab_size and performing softmax across vocabulary
        self.probs = T.nnet.softmax(logit.reshape([logit.shape[0]*logit.shape[1], logit.shape[2]])) #dim(probs) = (#timesteps*#samples) x vocab_size
        #Building loss function
        self.build_loss()
        
        self._outputs = [self.probs]

    def build_loss(self):
        # TODO: Remove the beg of sentence marker
        x_flat = self.x[1:].flatten() #x_flat: a linear array with size #timesteps*#samples
        x_flat_idx = T.arange(x_flat.shape[0]) * self.vocab.vocab_size + x_flat
                                    
        self._loss = -T.log(self.probs.flatten()[x_flat_idx])
        self._loss = self._loss.reshape([self.x.shape[0]-1, self.x.shape[1]])  # -1 is for removing beg of sent marker
        self._loss = (self._loss * self.x_mask[1:]).sum(0)

    def predictions(self):
        pass

    def targets(self):
        pass

    def loss(self):
        return self._loss.mean()
    
    def log_probs(self):
        # TODO: Make it better?
        return self._loss
    
    def outputs(self):
        return self._outputs
    
    def inputs(self):
        return [self.x, self.x_mask]
    
    def params(self):
        return self._params

    def load_params(self, params):
        for param_name in self.params():
            try:
                self._params[param_name].set_value(params[param_name])
            except KeyError:
                logger.error('Value for the model parameter %s is not found!' % param_name)

    def prepare_train_input(self, batch, max_length=None):
        # setting maxlen to length of longest sample
        if max_length is not None:
            # select the minimum of the max_length, or the length of the longest sample subject to max_length constraint
            try:
                max_length = min(max_length, max([len(sample) for sample in batch if len(sample) <= max_length]))
            except ValueError:
                # If there is no sample in the batch which is less than maxlength
                return None
        else:
            max_length = max([len(sample) for sample in batch]) 
        
        # converting to tokens
        inp = [[self.vocab.get_index(token) for token in sample] + [self.vocab.get_index(self.vocab.eos)] for sample in batch if len(sample) <= max_length]
        
        # adding eos token
        max_length = max_length + 1
        
        # preparing mask and input
        inp_mask = np.array([[1.] + [1.]*len(inp_instance[:max_length]) + [0.]*(max_length-len(inp_instance)) for inp_instance in inp], dtype='float32').transpose()
        inp = np.array([ [-1] + inp_instance[:max_length]  + [0.]*(max_length-len(inp_instance)) for inp_instance in inp], dtype='int64').transpose()
        return inp, inp_mask
