
from __future__ import absolute_import

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from collections import OrderedDict

from .model import Model
from ..nn.layers.embeddings import Embeddings
from ..nn.layers.gru import GRU, ConditionalGRU
from ..nn.layers.dense import Dense
from ..nn.activations import Activation


class RNNUnidirectionalEncDec(Model):
    
    def __init__(self, hyperparams, encoder_vocab, decoder_vocab):
        self.hyperparams = hyperparams
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        # Preparing and Initializing Network Weights & Biases
        self.setup()
    

    # TODO: Loading and storing params
    def setup(self):
        """
        Setup the shared variables and model components
        """
        self._params = OrderedDict()
        
        # Encoder embeddings
        self.encoder_embeddings = Embeddings('encoder_emb', self.hyperparams.encoder_vocab_size, self.hyperparams.encoder_emb_dim)
        self._params.update(self.encoder_embeddings.params())
        
        # Decoder embeddings
        self.decoder_embeddings = Embeddings('decoder_emb', self.hyperparams.decoder_vocab_size, self.hyperparams.decoder_emb_dim, add_bos=True)
        self._params.update(self.decoder_embeddings.params())
        
        
        # Encoder Layer
        if self.hyperparams.rnn_cell == 'gru':
            # left-to-right rnn
            self.encoder_rnn_layer_l2r = GRU(name='encoder_gru0_l2r', in_dim=self.hyperparams.encoder_emb_dim, num_units=self.hyperparams.encoder_units)
            self._params.update(self.encoder_rnn_layer_l2r.params())
        elif self.hyperparams.rnn_cell == 'lstm':
            raise NotImplementedError
        else:
            logger.error("Invalid RNN Cell Type:" + self.hyperparams.rnn_cell)
        
        # Transform to prepare init state of decoder
        self.decoder_init_transform = Dense(name='decoder_init_transform', in_dim=self.hyperparams.encoder_units, num_units=self.hyperparams.decoder_units, activation=Activation.tanh)
        self._params.update(self.decoder_init_transform.params())
        
        # Decoder Layer
        if self.hyperparams.rnn_cell == 'gru':
            self.decoder_rnn_layer = ConditionalGRU(name='decoder_gru0', in_dim=self.hyperparams.decoder_emb_dim, num_units=self.hyperparams.decoder_units, context_dim=self.hyperparams.encoder_units)
            self._params.update(self.decoder_rnn_layer.params())
        elif self.hyperparams.rnn_cell == 'lstm':
            raise NotImplementedError
        else:
            logger.error("Invalid RNN Cell Type:" + self.hyperparams.rnn_cell)
            
        # Read out words
        
        self.decoder_state_transform = Dense(name='decoder_state_transform', in_dim=self.hyperparams.decoder_units, num_units=self.hyperparams.decoder_emb_dim, activation=Activation.linear)
        self._params.update(self.decoder_state_transform.params())
        
        self.prev_emb_transform = Dense(name='prev_emb_transform', in_dim=self.hyperparams.decoder_emb_dim, num_units=self.hyperparams.decoder_emb_dim, activation=Activation.linear)
        self._params.update(self.prev_emb_transform.params())

        self.encoder_context_transform = Dense(name='encoder_context_transform', in_dim=self.hyperparams.encoder_units, num_units=self.hyperparams.decoder_emb_dim, activation=Activation.linear)
        self._params.update(self.encoder_context_transform.params())
        
        self.word_probs_transform = Dense(name='word_probs_transform', in_dim=self.hyperparams.decoder_emb_dim, num_units=self.hyperparams.decoder_vocab_size, activation=Activation.linear)
        self._params.update(self.word_probs_transform.params())

        # DEBUG
        #for k, v in self._params.iteritems():
        #    print k, v.get_value(), v.get_value().shape
        
    def build(self):
        self.trng = RandomStreams(1234)
        
        # dim(x) = (input_time_steps, num_samples)
        self.x = T.matrix('x', dtype='int64')
        # dim(x_mask) = (input_time_steps, num_samples)
        self.x_mask = T.matrix('x_mask', dtype='float32')
        # dim(y) = (output_time_steps, num_samples)
        self.y = T.matrix ('y', dtype='int64')
        # dim(y_mask) = (output_time_steps, num_samples)
        self.y_mask = T.matrix('y_mask', dtype='float32')
        
        # get source word embeddings
        enc_emb = self.encoder_embeddings.Emb[self.x.flatten()]
        # dim(x) = timesteps x samples
        enc_emb = enc_emb.reshape([self.x.shape[0], self.x.shape[1], self.hyperparams.encoder_emb_dim])
        
        # get decoder init state
        encoder_outputs = self.encoder_rnn_layer_l2r.build(enc_emb,  self.x_mask)[0] 
        last_encoder_output = encoder_outputs[-1]  # This will be the context at every input step
        
        # transform encoder output to get decoder init
        decoder_init = self.decoder_init_transform.build(last_encoder_output)
        
        # input
        # TODO: remove embedding shifting?
        dec_emb = self.decoder_embeddings.Emb[self.y.flatten()]
        dec_emb = dec_emb.reshape([self.y.shape[0], self.y.shape[1], self.hyperparams.decoder_emb_dim])
        dec_emb_shifted = T.zeros_like(dec_emb)
        dec_emb_shifted = T.set_subtensor(dec_emb_shifted[1:], dec_emb[:-1])
        
        # Building the RNN layer
        decoder_outputs = self.decoder_rnn_layer.build(x=dec_emb_shifted, x_mask=self.y_mask, c=last_encoder_output, h_init=decoder_init)[0]  # Only one output, hidden states at every time step
        
        # context with new axis. condition the output on encoder context as well.
        # TODO: remove axis adding?
        context = last_encoder_output[None, :, :]
        
        # dim(proj_h) = #timesteps x #samples x #num_units
        
        # Computing word probabilities
        logit_decoder_rnn = self.decoder_state_transform.build(decoder_outputs)  # dim(logit_rnn) = #timesteps x #samples x #emb_dim
        logit_prev_emb = self.prev_emb_transform.build(dec_emb_shifted)   # dim(logit_prev) = #timesteps x #samples x #emb_dim
        logit_enc_context = self.encoder_context_transform.build(context)
        logit = self.word_probs_transform.build(Activation.tanh(logit_decoder_rnn + logit_prev_emb + logit_enc_context)) # dim(logit) = #timesteps x #samples x #vocab_size
        
        # reshaping logit as (#timesteps*#samples) x vocab_size and performing softmax across vocabulary
        self.probs = T.nnet.softmax(logit.reshape([logit.shape[0]*logit.shape[1], logit.shape[2]])) #dim(probs) = (#timesteps*#samples) x vocab_size
        #Building loss function
        self.build_loss()
        
        self._outputs = [self.probs]

    def build_loss(self):
        # TODO: Make it better?
        y_flat = self.y.flatten() #x_flat: a linear array with size #timesteps*#samples
        y_flat_idx = T.arange(y_flat.shape[0]) * self.hyperparams.decoder_vocab_size + y_flat
                                    
        self._loss = -T.log(self.probs.flatten()[y_flat_idx])
        self._loss = self._loss.reshape([self.y.shape[0], self.y.shape[1]])
        self._loss = (self._loss * self.y_mask).sum(0)    

    def loss(self):
        return self._loss.mean()
    
    def log_probs(self):
        # TODO: Make it better?
        return self._loss
    
    def outputs(self):
        return self._outputs
    
    def inputs(self):
        return [self.x, self.x_mask, self.y, self.y_mask]
        
    def params(self):
        return self._params
    

    def prepare_input(self, batch, max_length=None):
        # setting maxlen to length of longest sample
        max_length_input = max([len(sample[0]) for sample in batch]) 
        max_length_target = max([len(sample[1]) for sample in batch])
        
        # adding end of sentence marker
        inp = [[self.encoder_vocab.get_index(token) for token in sample[0]] + [self.encoder_vocab.get_index(self.encoder_vocab.eos)] 
               for sample in batch]
        target = [[self.decoder_vocab.get_index(token) for token in sample[1]] + [self.decoder_vocab.get_index(self.decoder_vocab.eos)] 
               for sample in batch]
        max_length_input += 1
        max_length_target += 1
        
        # preparing mask and input
        inp_mask = np.array([[1.]*len(inp_instance[:max_length_input]) + [0.]*(max_length_input-len(inp_instance)) for inp_instance in inp], dtype='float32').transpose()  
        inp = np.array(               [inp_instance[:max_length_input]  + [0.]*(max_length_input-len(inp_instance)) for inp_instance in inp], dtype='int64').transpose()
        target_mask = np.array([[1.]*len(target_instance[:max_length_target]) + [0.]*(max_length_target-len(target_instance)) for target_instance in target], dtype='float32').transpose()  
        target = np.array(               [target_instance[:max_length_target]  + [0.]*(max_length_target-len(target_instance)) for target_instance in target], dtype='int64').transpose()
       
        return inp, inp_mask, target, target_mask

