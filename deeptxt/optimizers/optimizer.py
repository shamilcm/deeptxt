import abc, six

@six.add_metaclass(abc.ABCMeta)
class Optimizer:
    
    @abc.abstractmethod
    def minimize(self):
        pass
    
