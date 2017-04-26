from __future__ import absolute_import


## read dictionary file and prepare vocabulary
class VocabManager:
    """
    Initialize the vocabulary manager object.
    
    #Arguments
        vocab_path: path to the vocabulary file, one token per line, sorted based on count (descending).
        vocab_size: number of tokens in the vocabulary to retain
        encoding: encoding to use while reading the file (default: utf-8)
    """

    def __init__(self, vocab_path, vocab_size, encoding='utf-8'):
        # Setting the instance variables
        self.vocab_path = vocab_path
        self.vocab_size = vocab_size + 2  # +2 for <unk> and </s> 
        self.token_to_index_dict = dict() 
        self.index_to_token_dict = dict()

        # Setting end of symbol token
        self.eos = '</s>'.decode(encoding)
        self.token_to_index_dict[self.eos] = 0
        self.index_to_token_dict[1] = self.eos
        
        # Setting unknown token in dictionary
        self.unk = '<unk>'.decode(encoding)
        self.token_to_index_dict[self.unk] = 1
        self.index_to_token_dict[0] = self.unk
        

        # Loop through the vocabulary.
        index = 2
        with open(vocab_path) as fvocab:
            for line in fvocab:
                if not line:
                    continue
                token,_ = line.decode(encoding).strip().split()
                self.token_to_index_dict[token] = index
                self.index_to_token_dict[index] = token
                index += 1
                if index >= self.vocab_size:
                    break
    
    def get_token(self, index):
        """
        Gets the token correspodning to the index.
        
        :param index: index to the vocabulary
        """
        try:
            return self.index_to_token_dict[index]
        except KeyError:
            logger.error("Invalid index to vocabulary!")
        
    def get_index(self, token):
        """
        Gets the index corresponding to the token.
        
        :param token: the token whose index is to be retrieved
        """
        try:
            return self.token_to_index_dict[token]
        except KeyError:
            return self.token_to_index_dict[self.unk]
                