import abc, six

@six.add_metaclass(abc.ABCMeta)
class Layer:
    
    @property
    def params(self):
        pass
