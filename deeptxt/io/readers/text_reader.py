import gzip
import itertools #TODO: import only islice to improve speed?

class TextReader:
    def __init__(self, dataset_path,  batchsize=1,   encoding='utf-8', max_length=None):
        """
        Initialize a text iterator object to read text file batch by batch
        """
        self.dataset_path = dataset_path
        self.batchsize = batchsize
        self.encoding = encoding
        self.max_length = max_length
        ## TODO: Shuffling the data !!!
        # Reading from gzip file
        if self.dataset_path.endswith('.gz'):
            self.dataset = gzip.open(self.dataset_path,'r')
        else:
            self.dataset = open(self.dataset_path, 'r')
    
    def __iter__(self):
        """
        Default method for iterator object
        """
        return self
    
    def reset(self):
        """
        Resets the file pointer back to the beginning
        """
        self.dataset.seek(0)
        
    def next(self):
        """
        Default method for iterator object
        :return: returns the batch
        """
        if self.max_length == None:
            lines = list(itertools.islice(self.dataset, self.batchsize))   
            #print [line.split() for line in lines]
            if lines == []:
                self.reset()
                raise StopIteration
            samples = [[token for token in line.decode(self.encoding).strip().split()] for line in lines]
        else:
            samples = []
            while True:
                line = self.dataset.readline()
                if line == '': # check EOF
                    if samples == []:
                        self.reset()
                        raise StopIteration
                    else:
                        return samples
                tokens = line.decode(self.encoding).strip().split()
                if len(tokens) > self.max_length:
                    continue
                samples.append(tokens)
                if len(samples) >= self.batchsize:
                    break
        return samples