import gzip
import itertools # TODO: import only islice to improve speed?

class ParallelTextReader:
    def __init__(self, source_dataset_path, target_dataset_path,  batchsize=1,   encoding='utf-8', source_max_length=None, target_max_length=None, num_batches_in_cache=20, sort_by_target=True):
        """
        Initialize a text iterator object to read text file batch by batch
        """

        # TODO: caching to improve speed
        self.source_dataset_path = source_dataset_path
        self.target_dataset_path = target_dataset_path
        self.batchsize = batchsize
        self.encoding = encoding
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

        self.cache_size = num_batches_in_cache
        self.cache = []
        self.sort_by_target = sort_by_target

        ## TODO: Shuffling the data !!!

        # Opining dataset
        def dataset_open(dataset_path):
            if dataset_path.endswith('.gz'):
                dataset = gzip.open(dataset_path,'r')
            else:
                dataset = open(dataset_path, 'r')
            return dataset

        self.source_dataset = dataset_open(self.source_dataset_path)
        self.target_dataset = dataset_open(self.target_dataset_path)

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
        self.source_dataset.seek(0)
        self.target_dataset.seek(0)

    def get_batch(self):
        samples = []
        while True:
            src_line = self.source_dataset.readline()
            trg_line = self.target_dataset.readline()
            if src_line == '' and trg_line == '': #Reached EOF
                if not samples:
                    self.reset()
                    return samples
                else:
                    return samples
            elif (src_line == '' and trg_line != '') or  (src_line != '' and trg_line == ''):
                print >> sys.stderr, "Number of lines in source and target datasets do not match!"
                sys.exit()

            src_tokens = src_line.decode(self.encoding).strip().split()
            trg_tokens = trg_line.decode(self.encoding).strip().split()

            # continue iteration if  source  max lengths
            # TODO: remove flags...
            #flag = 0
            if self.source_max_length != None and  len(src_tokens) > self.source_max_length:
                #flag = 1
                continue

            # continue iteration if target sentence exceeds maxlength
            if self.target_max_length != None and len(trg_tokens) > self.target_max_length:
                #if flag == 1:
                continue

            samples.append((src_tokens, trg_tokens))
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
            cache_samples = sorted(cache_samples, key=lambda x: len(x[1]), reverse=True)

        self.cache = [cache_samples[i:i+self.batchsize] for i in xrange(0, len(cache_samples), self.batchsize)]
        self.cache.reverse()

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


