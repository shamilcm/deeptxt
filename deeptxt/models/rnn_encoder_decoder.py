
from __future__ import absolute_import

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from collections import OrderedDict

from .model import Model
from ..nn.layers.embeddings import Embeddings
from ..nn.layers.dense import Dense
from ..nn.activations import Activation

import logging
from ..utils import logutils
logger = logging.getLogger(__name__)


class RNNEncoderDecoder(Model):

    def __init__(self, hyperparams, encoder_vocab, decoder_vocab, use_attention=False, encode_bidirectional=False, **kwargs):
        self._hyperparams = hyperparams
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.use_attention = use_attention
        self.encode_bidirectional = encode_bidirectional

        # hyperparams.encoder_vocab_size and hyperparams.decoder_vocab_size setting to max
        # TODO: Uncomment this and throw error
        #hyperparams.encoder_vocab_size = min(hyperparams.encoder_vocab_size, encoder_vocab.vocab_size)
        #hyperparams.decodeRr_vocab_size = min(hyperparams.decoder_vocab_size, decoder_vocab.vocab_size)

        # Preparing and Initializing Network Weights & Biases
        self.setup()

    def hyperparams(self):
        return self._hyperparams


    # TODO: Loading and storing params
    def setup(self):
        """
        Setup the shared variables and model components
        """
        self._params = OrderedDict()

        # Encoder embeddings
        self.encoder_embeddings = Embeddings('encoder_emb', self.encoder_vocab.vocab_size, self.hyperparams().encoder_emb_dim)
        self._params.update(self.encoder_embeddings.params())

        # Decoder embeddings
        self.decoder_embeddings = Embeddings('decoder_emb', self.decoder_vocab.vocab_size, self.hyperparams().decoder_emb_dim, add_bos=True)
        self._params.update(self.decoder_embeddings.params())

        ################
        # Encoder Layer
        ################
        # TODO: make a different class
        if self.hyperparams().rnn_cell == 'gru':
            from ..nn.layers.gru import GRU as RNN
        elif self.hyperparams().rnn_cell == 'lstm':
            raise NotImplementedError
        else:
            logger.error("Invalid RNN Cell Type:" + self.hyperparams().rnn_cell)

        self.encoder_rnn_layer_l2r = RNN(name='encoder_' + self.hyperparams().rnn_cell + '0_l2r', in_dim=self.hyperparams().encoder_emb_dim, num_units=self.hyperparams().encoder_units)
        self._params.update(self.encoder_rnn_layer_l2r.params())
        encoding_dim = self.hyperparams().encoder_units

        if self.encode_bidirectional == True:
            self.encoder_rnn_layer_r2l = RNN(name='encoder_' + self.hyperparams().rnn_cell + '0_r2l', in_dim=self.hyperparams().encoder_emb_dim, num_units=self.hyperparams().encoder_units)
            self._params.update(self.encoder_rnn_layer_r2l.params())
            encoding_dim = self.hyperparams().encoder_units * 2

        # Transform to prepare init state of decoder
        self.decoder_init_transform = Dense(name='decoder_init_transform', in_dim=encoding_dim, num_units=self.hyperparams().decoder_units, activation=Activation.tanh)
        self._params.update(self.decoder_init_transform.params())

        ################
        # Decoder Layer
        ###############
        # TODO: make a different class
        if self.hyperparams().rnn_cell == 'gru':
            if self.use_attention:
                from ..nn.layers.gru import ConditionalGRU_Attn as ConditionalRNN
            else:
                from ..nn.layers.gru import ConditionalGRU as ConditionalRNN
        elif self.hyperparams().rnn_cell == 'lstm':
            raise NotImplementedError
        else:
            logger.error("Invalid RNN Cell Type:" + self.hyperparams().rnn_cell)

        self.decoder_rnn_layer = ConditionalRNN(name='decoder_' + self.hyperparams().rnn_cell + '0', in_dim=self.hyperparams().decoder_emb_dim, num_units=self.hyperparams().decoder_units, context_dim=encoding_dim)
        self._params.update(self.decoder_rnn_layer.params())

        # Read out words

        self.decoder_state_transform = Dense(name='decoder_state_transform', in_dim=self.hyperparams().decoder_units, num_units=self.hyperparams().decoder_emb_dim, activation=Activation.linear)
        self._params.update(self.decoder_state_transform.params())

        self.prev_emb_transform = Dense(name='prev_emb_transform', in_dim=self.hyperparams().decoder_emb_dim, num_units=self.hyperparams().decoder_emb_dim, activation=Activation.linear)
        self._params.update(self.prev_emb_transform.params())

        self.encoder_context_transform = Dense(name='encoder_context_transform', in_dim=encoding_dim, num_units=self.hyperparams().decoder_emb_dim, activation=Activation.linear)
        self._params.update(self.encoder_context_transform.params())

        self.word_probs_transform = Dense(name='word_probs_transform', in_dim=self.hyperparams().decoder_emb_dim, num_units=self.decoder_vocab.vocab_size, activation=Activation.linear)
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
        encoder_emb = self.encoder_embeddings.Emb[self.x.flatten()]
        # dim(x) = timesteps x samples
        encoder_emb = encoder_emb.reshape([self.x.shape[0], self.x.shape[1], self.hyperparams().encoder_emb_dim])

        # get decoder init state
        self.encoder_outputs = self.encoder_rnn_layer_l2r.build(encoder_emb,  self.x_mask)[0]

        if self.encode_bidirectional == True:
            #x_rev = self.x[::-1]
            #x_mask_rev = self.x_mask[::-1]
            #encoder_emb_rev = self.encoder_embeddings.Emb[x_rev.flatten()e
            #encoder_emb_rev = encoder_emb_rev.reshape([x_rev.shape[0], x_rev.shape[1], self.hyperparams().encoder_emb_dim])
            encoder_emb_rev = encoder_emb[::-1]
            encoder_outputs_rev = self.encoder_rnn_layer_r2l.build(encoder_emb_rev,  self.x_mask[::-1])[0]
            self.encoder_outputs = T.concatenate([self.encoder_outputs, encoder_outputs_rev[::-1]], axis=self.encoder_outputs.ndim - 1)

        #self.debug = self.encoder_outputs.shape

        if self.use_attention == True:
            # use unweighted mean of encoder_outputs to initialize the decoder
            encoder_output = (self.encoder_outputs * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]
        else:
            encoder_output = self.encoder_outputs[-1]

        # transform encoder output to get decoder init
        self.decoder_init = self.decoder_init_transform.build(encoder_output)

        # input
        # TODO: remove embedding shifting?
        dec_emb = self.decoder_embeddings.Emb[self.y.flatten()]
        dec_emb = dec_emb.reshape([self.y.shape[0], self.y.shape[1], self.hyperparams().decoder_emb_dim])

        # Building the RNN layer
        if self.use_attention == True:
            self.decoder_outputs, context, alphas = self.decoder_rnn_layer.build(x=dec_emb, x_mask=self.y_mask,
                                       C=self.encoder_outputs, C_mask=self.x_mask, h_init=self.decoder_init)
            # each of the above outputs is of size #decoder_timesteps x #samples x respective_dim

        else:
            # Only one output, hidden states at every time step
            self.decoder_outputs = self.decoder_rnn_layer.build(x=dec_emb, x_mask=self.y_mask,
                                                       c=encoder_output,  h_init=self.decoder_init)[0]
            # context with new axis. condition the output on encoder context as well.
            # TODO: remove axis adding?
            context = encoder_output[None, :, :]

        # dim(decoder_output) = #timesteps x #samples x #num_units

        # Computing word probabilities
        logit_decoder_rnn = self.decoder_state_transform.build(self.decoder_outputs)  # dim(logit_rnn) = #timesteps x #samples x #emb_dim
        logit_prev_emb = self.prev_emb_transform.build(dec_emb)   # dim(logit_prev) = #timesteps x #samples x #emb_dim
        logit_enc_context = self.encoder_context_transform.build(context) # dim(context) = #timesteps x #samples x context_dim
        logit = self.word_probs_transform.build(Activation.tanh(logit_decoder_rnn + logit_prev_emb + logit_enc_context)) # dim(logit) = #timesteps x #samples x #vocab_size

        # removing the extra timestep due to the adding of <bos> for target
        #logit = logit[:-1]
        # to make it the same size as the reference input, removing an extra timestep
        # that occured because of adding <bos> in the beginning
        #self.logit_2d = logit.reshape([(logit.shape[0])*logit.shape[1], logit.shape[2]])
        # reshaping logit as (#timesteps*#samples) x vocab_size and performing softmax across vocabulary
        self.probs = T.nnet.softmax(logit.reshape([logit.shape[0]*logit.shape[1], logit.shape[2]]))
        #self.probs = T.nnet.softmax(self.logit_2d) #dim(probs) = (#timesteps*#samples) x vocab_size

        #self.probs = self.probs.reshape([logit.shape[0], logit.shape[1], logit.shape[2]])




        # required for test time
        self._outputs = [self.probs]

        # Required for the loss function
        self._predictions = self.probs.reshape([logit.shape[0], logit.shape[1], logit.shape[2]])[:-1]  # remove last row because of extra <bos> adding oen additional timestep
        self._targets = self.y[1:]
        self._targets_mask = self.y_mask[1:]

        # Building loss function
        self.build_loss()



    def build_loss(self):
        # TODO: Make it better?
        # y[0]  is bos, remove it to calculate loss
        # y_flat = self.y[1:].flatten() #x_flat: a linear array with size #timesteps*#samples
        # y_flat_idx = T.arange(y_flat.shape[0]) * (self.decoder_vocab.vocab_size) + y_flat

        # self._loss = -T.log(self.probs.flatten()[y_flat_idx])
        # self._loss = self._loss.reshape([self.y.shape[0]-1, self.y.shape[1]])
        # self._loss = (self._loss * self.y_mask[1:]).sum(0)
        # self._loss = self._loss.mean()

        ### Method 2
        #self._loss = T.nnet.categorical_crossentropy(self.probs, y_flat)

        ### Method 3
        #y_flat = self.y[1:].flatten() #x_flat: a linear array with size #timesteps*#samples
        #y_flat_idx = T.arange(y_flat.shape[0]) * (self.decoder_vocab.vocab_size) + y_flat
        #log_z = T.log(T.exp(self.logit_2d).sum(1))
        #self._loss = log_z - self.logit_2d.flatten()[y_flat_idx]
        #self._loss = self._loss.reshape([self.y.shape[0]-1, self.y.shape[1]])
        #self._loss = (self._loss * self.y_mask[1:]).sum(0)
        # self._loss = self._loss.mean()

        ### Method 4
        from ..objectives.losses import categorical_crossentropy
        targets, targets_mask = self.targets()
        self._loss = categorical_crossentropy(targets=targets, predictions=self.predictions(), mask=targets_mask)

        ## Section to remove after debugging
        #try:
        #    self.debug = self._loss[1]
        #    self._loss = self._loss[0]
        #except:
        #    pass

        #y_flat = targets.flatten()
        #y_flat_idx = T.arange(y_flat.shape[0]) * self._predictions.shape[-1] + y_flat

        #self._loss = -T.log(self._predictions.flatten()[y_flat_idx])
        #self._loss = self._loss.reshape([targets.shape[0], targets.shape[1]])
        #self._loss = (self._loss * targets_mask).sum(0)
        #self._loss = self._loss.mean()


    def build_sampler(self, sampling=True):
        initializer_input = [self.x, self.x_mask]
        if self.use_attention == True:
            initializer_output = [self.encoder_outputs, self.x_mask, self.decoder_init]
        else:
            initializer_output = [self.encoder_outputs, self.decoder_init]
        self.initializer = theano.function(initializer_input, initializer_output)

        sampler_input = [self.y, self.y_mask] + initializer_output

        if sampling == True:
            # sample a word from the output softmax, instead of selecting argmax
            next_token_index = self.trng.multinomial(pvals=self.probs).argmax(1)  # multinomial will represent 1 hot representation of the selected sample
        else:
            next_token_index = T.argmax(self.probs, axis=1)

        sampler_output = [self.probs, next_token_index, self.decoder_outputs]
        self.sampler = theano.function(sampler_input, sampler_output)

    def sample(self, model_input,  num_samples=5):
        source, source_mask, target, target_mask = model_input
        # find if num_samples or source is fewer
        num_samples = np.minimum(num_samples, source.shape[1])
        # TODO: replace by random sampling:
        source = source[:,0:num_samples]
        source_mask = source_mask[:,0:num_samples]
        target = target[:, 0:num_samples]
        target_mask = target_mask[:, 0:num_samples]

        samples = []
        for sample_index in xrange(num_samples):
            hypothesis = self.encode_decode([ source[:, sample_index:sample_index+1], source_mask[:, sample_index:
                sample_index+1] ])
            hypothesis_sent = ' '.join(hypothesis)

            source_sent = ' '.join([self.encoder_vocab.get_token(index) for index in source[:, sample_index] if index != self.encoder_vocab.get_index(self.encoder_vocab.eos)])

            #hypothesis_sent = ' '.join([self.decoder_vocab.get_token(index) for index in target[:, sample_index] if index != self.decoder_vocab.get_index(self.decoder_vocab.eos)])

            target_sent = ' '.join([self.decoder_vocab.get_token(index) for index in target[:, sample_index] if index != -1 and index != self.decoder_vocab.get_index(self.decoder_vocab.eos)])
            # TODO: change sample to dictionary, and in trainer display all keys and values
            samples.append(OrderedDict({'SRC': source_sent, 'HYP': hypothesis_sent, 'REF': target_sent}))

        return samples

    def encode_decode(self, test_input, max_length=50):
        encoding = self.initializer(*test_input)
        init_input = [np.array([[-1]]), np.array([[1.]], dtype='float32')] + encoding

        probs, next_token_index, decoder_outputs = self.sampler(*init_input)

        hypothesis = []
        hyp_length = 0
        while(next_token_index[0] != self.decoder_vocab.get_index(self.decoder_vocab.eos)):
            hypothesis.append(self.decoder_vocab.get_token(next_token_index[0]))
            hyp_length += 1
            # This is if next_token_index is a scalar, i.e. when only one column is passed as test_input
            # [None] adds a new axis
            next_input = [next_token_index[None], np.array([[1.]], dtype='float32')] + encoding
            probs, next_token_index, decoder_outputs = self.sampler(*next_input)
            if hyp_length == max_length:
                break

        return hypothesis

    def loss(self):
        return self._loss

    def predictions(self):
        return self._predictions

    def targets(self):
        return self._targets, self._targets_mask

    def log_probs(self):
        # TODO: Make it better?
        return self._loss

    def outputs(self):
        return self._outputs

    def inputs(self):
        return [self.x, self.x_mask, self.y, self.y_mask]

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
        source_mask = np.array([[1.]*len(inp_instance[:max_length_input]) + [0.]*(max_length_input-len(inp_instance)) for inp_instance in inp], dtype='float32').transpose()
        source = np.array(               [inp_instance[:max_length_input]  + [0.]*(max_length_input-len(inp_instance)) for inp_instance in inp], dtype='int64').transpose()

        # taret preparation with -1 (beginning of sentence) row upfront
        target_mask = np.array([[1.] + [1.]*len(target_instance[:max_length_target]) + [0.]*(max_length_target-len(target_instance)) for target_instance in target], dtype='float32').transpose()
        target = np.array([[-1] + target_instance[:max_length_target]  + [0.]*(max_length_target-len(target_instance)) for target_instance in target], dtype='int64').transpose()
        return source, source_mask, target, target_mask

    def prepare_test_input(self, sentence):
        tokens = sentence.split() + [self.encoder_vocab.eos]
        inp = np.array([[self.encoder_vocab.get_index(token) for token in tokens ]], dtype='int64')
        inp_mask = np.array([[1.]*(len(tokens))], dtype='float32')
        return inp.transpose(), inp_mask.transpose()

    def prepare_sampler_input(self, prev_token_list, decoder_states, initializer_output=None):
        num_samples = len(prev_token_list)

        assert len(prev_token_list) == decoder_states.shape[0], 'Number of previous tokens, and number of decoder states should match'
        y =  np.array([[self.decoder_vocab.get_index(token) for token in prev_token_list]])
        y_mask = np.array([[1.]*num_samples],dtype='float32')

        if initializer_output:
            encoder_outputs = initializer_output[0]
            encoder_outputs = np.tile(encoder_outputs, (1, num_samples, 1))


            # preparing decoder init
            #decoder_state = initializer_output[-1]
            #decoder_state = np.tile(decoder_state, (num_samples, 1))

            #print "DEBUG", decoder_state.shape
            #print "DEB", decoder_states.shape

            if len(initializer_output) == 3:

                # preparing xmask
                x_mask = initializer_output[1]  # for attention-based models
                x_mask = np.tile(x_mask, (1, num_samples))

                encoder_info = [encoder_outputs, x_mask] 


            elif len(initializer_output) == 2:
                encoder_info = [encoder_outputs]

            rets = [y, y_mask] + encoder_info + [decoder_states]
            return rets
        else:
            return [y, y_mask]

