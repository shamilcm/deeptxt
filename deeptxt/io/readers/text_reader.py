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
    '''
    def __iter__(self):
        """
        Default method for iterator object
        """
        return self
    '''


    def reset(self):
        """
        Resets the file pointer back to the beginning
        """
        self.dataset.seek(0)

    def get_batch(self):
        """
        Default method for iterator object
        :return: returns the batch
        """
        samples = []
        while True:
            line = self.dataset.readline()
            if line == '': # check EOF
                if not samples:
                    self.reset()
                    return samples
                else:
                    return samples
            tokens = line.decode(self.encoding).strip().split()
            if len(tokens) > self.max_length:
                continue
            samples.append(tokens)
            if len(samples) >= self.batchsize:
                break
        return samples


    def fill_cache(self):
        cache_samples = []
        eof = False
        for i in xrange(self.cache_size):
            batch_samples = self.get_batch()

            # empty batch_samples indicates end of file
            if not batch_samples:
                eof = True
                break
            else:
                cache_samples += batch_samples

        # sort the batches by length, to make training more efficient
        # varying lengths may create large and sparse matrices for all updates.
        if self.sort_by_target == True:
            cache_samples = sorted(cache_samples, key=lambda x: len(x[1]))

        self.cache = [cache_samples[i:i+self.batchsize] for i in xrange(0, len(cache_samples), self.batchsize)]
        
        # if eof is within cache, i.e. last of the batches have been read. Then None is put at the beginning
        if eof == True:
            self.cache = [None] + self.cache



    def next(self):
        """
        Default method for iterator object
        :return: returns the batch
        """
        if not self.cache:
            self.fill_cache()

        samples = self.cache.pop()
        return samples