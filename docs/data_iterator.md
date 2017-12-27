# Data Iterators

Data iteratotrs act as an mediator between the data set and models.

```
                    IDataIteerator               FeatureType
                            |                        |
                            ---------     ------------
                                     |    |
                                CustomDataIterator
```

## Interface
[IDataIterator](../src/interfaces/data_iterator.py)

## What is expected?

We are following Tensorflow dataset APIs from [here](https://www.tensorflow.org/programmers_guide/datasets)

Once again do some magic ;) and implement the interface.

TODO: add how it is done on CoNLL data
