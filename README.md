## deeptxt

A neural machine translation and recurrent language modelling toolkit, currently supporting Theano backend. 

deeptxt follows the same architecture as that used by [Nematus](https://github.com/rsennrich/nematus)  and [DL4MT-Tutorial](https://github.com/nyu-dl/dl4mt-tutorial).

### Requirements

* Theano, Pythhon and Numpy 


### Training


```
./train -c /path/to/config/file -o /path/to/output/dir -d cuda
```


### Testing
```
./infer -m /path/to/model -v /path/to/source_vocab /path/to/target_vocab < input-file > output-file
```
