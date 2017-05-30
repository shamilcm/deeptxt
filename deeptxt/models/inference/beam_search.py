import numpy as np
import collections
import copy

BeamObject = collections.namedtuple('BeamObject', 'hypothesis score state')


class BeamSearch:
    def __init__(self, beam_size, model, sample, max_length=100):
        self.beam_size = beam_size
        self.max_length = max_length

        self._model = model
        self._input = self._model.prepare_test_input(sample)
        self.search()

    def search(self):

        init_input = []
        if hasattr(self._model, 'initializer'):
            self.encoding = self._model.initializer(*self._input)
            init_word = np.array([[-1]])
            init_word_mask = np.array([[1.]], dtype='float32')
            init_input = [init_word, init_word_mask] + self.encoding

        probs, next_token_index, decoder_outputs = self._model.sampler(*init_input)

        logprobs = np.log(probs)
        vocab_size = logprobs.shape[1]
        candidates = np.argpartition(logprobs, -self.beam_size, axis=None)[-self.beam_size:]

        state_dict = {'F0':copy.copy(decoder_outputs[0,0])}

        self.active_beam = [ BeamObject(hypothesis=[self._model.decoder_vocab.get_token(index)],
                                        score=logprobs.flatten()[index],
                                        state=state_dict) for index in candidates]

        self.completed_beam = []


        active_beam_size = len(self.active_beam)

        steps = 0
        while True:
            steps += 1

            # popping beams which have been finished and putting it in the completed beams
            last_tokens = []
            last_states = dict()
            pop_indices = dict()

            for beam_index in xrange(len(self.active_beam)):

                beam_object = self.active_beam[beam_index]
                last_token = beam_object.hypothesis[-1]


                if last_token == self._model.decoder_vocab.eos:
                    # remove the end of sentence token and append to the completed beam
                    self.completed_beam.append(beam_object)
                    pop_indices[beam_index] = True
                    active_beam_size = active_beam_size - 1
                else:
                    last_tokens.append(last_token)
                    for k,v in beam_object.state.iteritems():
                        if k not in last_states:
                            last_states[k] = []
                        last_states[k].append(v)



            self.active_beam = [beam_object for beam_index, beam_object in enumerate(self.active_beam) if beam_index not in pop_indices]


            # stop beam search, if nothin exists in active beam
            if active_beam_size == 0:
                break

            #for bo in self.active_beam:

            # TODO : adjust encoding preparation to avoid repeating self.encoding
            # TODO: ensemble
            last_states['F0'] = np.array(last_states['F0'])
            sampler_input = self._model.prepare_sampler_input(last_tokens, last_states['F0'], self.encoding)

            probs, next_token_index, decoder_outputs = self._model.sampler(*sampler_input)

            #print probs

            logprobs = np.log(probs)
            hypscores = np.array([[self.active_beam[beam_index].score] for beam_index in xrange(len(self.active_beam))])
            logprobs = logprobs + hypscores #broadcasting hypscores [dim: #beamsize x 1] with logprobs [ dim: #beamsize x #vocabsize]
            candidates = np.argpartition(logprobs, -active_beam_size, axis=None)[-active_beam_size:]

            beam_indices = candidates/vocab_size
            vocab_indices = candidates % vocab_size

            self.active_beam = [ BeamObject(hypothesis=self.active_beam[beam_index].hypothesis +
                                                [self._model.decoder_vocab.get_token(vocab_index)],
                                            score=logprobs[beam_index][vocab_index],
                                            state={'F0':copy.copy(decoder_outputs[0,beam_index])})
                                for beam_index, vocab_index in zip(beam_indices, vocab_indices)]

            # stop beam search if maximum steps have reached
            if steps >= self.max_length:
                self.completed_beam += self.active_beam
                break


        self.completed_beam = sorted(self.completed_beam, key=lambda x: x.score, reverse=True)


    def best_hypothesis_sentence(self):
        if self.completed_beam[0].hypothesis[-1] == '</s>':
            return ' '.join(self.completed_beam[0].hypothesis[:-1])
        else:
            return ' '.join(self.completed_beam[0].hypothesis)


    def best_hypothesis_score(self):
        return self.completed_beam[0].score

    def nbest(self):
        return self.completed_beam

    def moses_formatted_nbest(self, encoding='utf-8'):
        raise NotImplementedError
