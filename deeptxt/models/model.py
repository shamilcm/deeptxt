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
    def prepare_input(self):
        pass
    
    @abc.abstractmethod
    def inputs(self):
        pass
    
    @abc.abstractmethod
    def outputs(self):
        pass
    
    @abc.abstractmethod
    def params(self):
        pass