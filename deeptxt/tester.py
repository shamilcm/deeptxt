import theano

class Tester:
    def __init__(self, data_reader, model, metric=None, save_history=False):
        self.model = model
        self.f_loss = theano.function(model.inputs(), model.loss())
        self.data_reader = data_reader
        self.metric = metric
        self.save_history = save_history
        self._history = []
    
    def history(self):
        return self._history
    
    def compute_loss(self):
        test_loss_sum = 0.0
        num_batches = 0
        
        for minibatch in self.data_reader:
            inp = self.model.prepare_input(minibatch)
            test_loss_sum += self.f_loss(*inp) 
            num_batches += 1
        
        avg_loss = test_loss_sum/num_batches
        
        if self.save_history == True:
            self.history.append(avg_loss)
        return avg_loss
    
    def compute_metric(self):
        if self.metric == None:
            return 0.0
        else:
            raise NotImplementedError