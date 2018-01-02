# Handling Dataset

Each business problem is unique so is the data set.
There is no one format fits all cases.

We are aware of this fact and we are not enforcing any particular data format,
however we had to use the available open data set to validate our models and
at the same time use data set that we extracted from our in-house/client
database on same models.

To accomplish this we came up with different preprocessor routines that
can handle different data sets and convert them into a intermediate format
on demand so that the data pipeline can be stream lined down the line.

Currently we have two such preprocessing routines:
- [CoNLL Preprocessor](../src/preprocessor/conll_data_preprocessor.py)
    - This follows the traditional CoNLL 2003 format,
    like the data [here](../conll_experiments/data/val.txt)
    - Text files are converted into CSV file as part of preprocessing.
    - Check out the configs [here](../conll_csv_experiments/config/config.ini)

- Patent Data Preprocessor
    - This one is for our internal client data, which is a CSV format with some headers
    - Each word, its positional informaation in the PDF and its label from database forms a row
    - Compared to CoNLL format, here we have more features that can be used for trianing
    - The positional features are `page number, x-position and y-position of the word`
    - Check out the configs here

After preprocessing, the files are expected to be operated with [Pandas](https://pandas.pydata.org/).

## Interface
[IPreprocessorInterface](../src/interfaces/preprocessor_interface.py)


## What is expected?

In short do some magic and fill this `PreprocessedDataInfo` class :)

- Loading Config
    - As matter of usage, it is designed to read both from `*.ini` file
    or from runtime while initializing the module
    - We are sticking with `*.ini`, as it is one time job for each data set
    - Assuming each data set will have its own configuration, a `experiment_folder/config/*.ini`
    is used to store and read data specific configuration
- Target Directories
    - We wish not to preprocess the data every time, so before storing the preprocessed
    data appropriate data folders are created and **over write** should be handled dynamically
- PReprocess the data


```
Raw Data ---> Preprocess the data
 ```


## Misc
 - The task we faced was kind of NER tagging, so you could see IOB used along side
 of the target lables