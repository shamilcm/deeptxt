import gzip
import itertools # TODO: import only islice to improve speed?

class ParallelTextReader:
    def __init__(self, source_dataset_path, target_dataset_path,  batchsize=1,   encoding='utf-8', source_max_length=None, target_max_length=None):
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

    def __iter__(self):
        """
        Default method for iterator object
        """
        return self

    def reset(self):
        """
        Resets the file pointer back to the beginning
        """
        self.source_dataset.seek(0)
        self.target_dataset.seek(0)

    def next(self):
        """
        Default method for iterator object
        :return: returns the batch
        """
        if self.source_max_length == None and self.target_max_length == None:
            src_lines = list(itertools.islice(self.source_dataset, self.batchsize))
            trg_lines = list(itertools.islice(self.target_dataset, self.batchsize))
            #print [line.split() for line in lines]
            if src_lines == [] and trg_lines == []:
                self.reset()
                raise StopIteration

            samples = [([src_token for src_token in src_line.decode(self.encoding).strip().split()],
                        [trg_token for trg_token in trg_line.decode(self.encoding).strip().split()])
                        for src_line, trg_line in zip(src_lines, trg_lines)]
        else:
            samples = []
            while True:
                src_line = self.source_dataset.readline()
                trg_line = self.target_dataset.readline()
                if src_line == '' and trg_line == '': #Reached EOF
                    if samples == []:
                        self.reset()
                        raise StopIteration
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

