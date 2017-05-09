
from __future__ import absolute_import
from collections import Counter

def write_vocab(dataset_path, out_vocab_path, encoding='utf-8'):
    """
        Prepares the vocabulary file for a dataset. It contains all unique tokens of file with its count, 
        in descending order of token count.

# Arguments:
        dataset_path: path to the text file to be read
        out_vocab_path: path to the dictionary file to be outputted
    """
    ctr = Counter()
    with open(dataset_path) as f:
        lcount = 1
        for line in f:
            tokens = line.decode(encoding).strip().split()
            ctr.update(tokens)
    with open(out_vocab_path,'w') as fvocab:
        for token, count in ctr.most_common():
            fvocab.write(token.encode(encoding) + ' ' + str(count) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-file', required=True, help='path to input file')
    parser.add_argument('-o','--output-vocab-file', required=True, help='path to output vocabulary file')
    args = parser.parse_args()

    write_vocab(args.input_file, args.output_vocab_file)

        
