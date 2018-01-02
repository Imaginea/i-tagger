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

Example: [CoNLL DataIterator](../src/data_iterators/csv_data_iterator.py)
## What is expected?

We are following Tensorflow dataset APIs from [here](https://www.tensorflow.org/programmers_guide/datasets)

Each data iterator has to tackle the feature generation from the data set,
and construct Tensorflow data iterators, such that Tensorflow estimatore can use them
in batches while training, validation and testing/predicting.

