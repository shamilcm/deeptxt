from __future__ import absolute_import
import abc, six

@six.add_metaclass(abc.ABCMeta)
class Model:

    @abc.abstractmethod
    def setup(self):
        pass
    
    @abc.abstractmethod
    def build(self):
        pass
    
    @abc.abstractmethod
    def prepare_train_input(self):
        pass
    
    @abc.abstractmethod
    def inputs(self):
        pass
    
    @abc.abstractmethod
    def outputs(self):
        pass

    @abc.abstractmethod
    def targets(self):
        pass
    
    @abc.abstractmethod
    def predictions(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def hyperparams(self):
        pass

    def build_sampler(self):
        pass

    def sample(self, model_input, num_samples):
        return [{'OUTPUT':'sampling is not implemented for this model!'}]

    # TODO: make this abstract
    def load_params(self):
        raise NotImplementedError

