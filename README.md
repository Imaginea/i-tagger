# Introduction
A simple and modular Tensorflow model development environment to handle
sequence-to-sequence models.

Developing models to solve a problem for a data set at hand,
requires lot of trial and error methods.
Which includes and not limited to:
- Preparing the ground truth or data set for training and testing
    - Collecting the data from online or open data sources
    - Getting the data from in-house or client database
- Pre-processing the data set
    - Text cleaning
    - NLP processing
    - Meta feature extraction etc.,
- Data iterators, loading and looping the data examples for model
while training and testing
    - In memory - All data is held in RAM and looped in batches on demand
    - Reading from the disk on demand in batches
    - Maitaining different feature sets (i.e number of features and its types) for the model
- Models
    - Maintaining different models for same set of features
    - Good visualizing and debugging environment/tools
    - Start and pause the training at will
- Model Serving
    - Load a particular model from the pool of available models for a
    particular data set

# Related Work
Below two Git repos got our attention:
- https://github.com/guillaumegenthial/sequence_tagging
- https://github.com/Franck-Dernoncourt/NeuroNER

Both of the projects are excellent in their own way, however they lack few
things like support for different dataset and models in a modular way,
which plays a key role in a customer facing project(s). Where nature of
the data changes as the project evolves.

# Problem Statement
 - To come up with an software architecture to try different models on
 different data set
 - Which should take care of:
    - Pre-processing the data
    - Preparing the data iterators for training, validation and testing
    for set of features and their types
    - Use a model that aligns with the data iterator and a feature type
    - Train the model in an iterative manner, with fail safe
    - Use the trained model to predict on new data
 - Keep the **model core logic independent** of the current architecture

# Solution or proposal

A few object-oriented principles are used in the python scripts for
ease of extensibility and maintenance.

**What we solved using this code?**
 - Top level results on open CoNLL dataset 2003
 - Extract information from patent documents for form filling, from historical
data entries from the Database records.

### Current Architecture

- [Handling Dataset and Preprocessing](docs/dataset.md)
- [Data iterators](docs/data_iterator.md)
    - Dataset may have one or more features like words,
characters, positional information of words etc.,
    - Extract those and convert word/characters to numeric ids, pad them etc.,
    - Enforces number of features and their types, so that set of models
      can work on down the line
- [Models](docs/models.md) should agree with data iterator features types and
make use of the available features to train the data


![](docs/images/i_tagger_architecture.png)


- **[Tensorflow Estimators](https://www.tensorflow.org/extend/estimators)** is used for training/evaluating/saving/restoring/predicting

![](docs/images/tf_estimators.png)

#### Directory Details

Each experiment should have its own copy of the data set.

Lets consider CoNLL data set, since it is provided as part this repo
- [conll_csv_experiments](conll_csv_experiments/)
    - config
        - config.ini # all one time config goes here
    - data
        - train.txt
        - test.txt
        - val.txt
    - preprocessed_data
        - train/
        - val/
        - test/
    - csv_data_iterator
        - bilstm_crf_v0
            - charembd_True_lr_0.001_lstmsize_2-48-32_wemb_48_cemb_32_outprob_0.5
            - config_2


#### Available Models:
- [Models](docs/models)
    - [Variable Length BiLSTM with CRF](docs/models/bilstm_crf_v0/BiLSTM_CRF_V0.md)


# Validation
 The whole package is tested on CoNLL data set for software integrity,
 and results are not tuned yet!

**[Check here for more details on how to test it on CoNLL data set.](conll_csv_experiments/README.md)**


![](docs/images/conll_tensorboard_results.png)


-------------------------------------------------------------------

# Setup

## Requirements:
- Python 3.5
- tensorflow-gpu r1.4
- spaCy
- tqdm
- tmux
- overrides


## How run on GPU server: (Imginea Specific)

```bash
#run following command for one time password verification
ssh-copy-id "rpx@172.17.0.5"

ssh rpx@172.17.0.5

# One time setup
tmux new -s your_name
export PATH=/home/rpx/anaconda3/bin:$PATH

### Note following environment is already setup, 
### no need to replicate unles you wanted different versions
conda create -n tensorflow-gpu python=3.5 anaconda
export LD_LIBRARY_PATH=/home/rpx/softwares/cudnn6/cuda/lib64:$LD_LIBRARY_PATH
source activate tensorflow-gpu
python --version

```

## Anaconda Environmnet setup: (General Users)

```bash
conda create -n tensorflow-gpu python=3.5 anaconda
source activate tensorflow-gpu
```

## Environment setup:
```bash
pip install tensorflow_gpu
pip install spacy
python -m spacy download en_core_web_md
pip install tqdm
pip install overrides
```

## Tmux (Imginea Specific)
```
cd ~/experiments/
mkdir your_name
cd your_name

git clone https://gitlab.pramati.com/imaginea-labs/i-tagger

```

**Day to day use**
```

tmux a -t your_name

### run only if you previous tmux session was closed completly
source activate tensorflow-gpu
export PATH=/home/rpx/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/rpx/softwares/cudnn6/cuda/lib64:$LD_LIBRARY_PATH

```

### Run Tests
```
cd path/to/i-tagger/

python -m unittest discover src/
```

### Learning Materials
- [Walk Through of Tensorflow APIs](notebooks/walk_through_of_tf_apis.ipynb)


# Authors
- Mageswaran Dhandapani <mageswaran.dhandapani@imaginea.com>
- Gaurish Thakkar <gaurish.thakkar@imaginea.com>
- Anil Kumar Reddy <anilkumar.reddy@imaginea.com>


TODOs:
- Remove all default params
- Tune the model for CoNLL dataset
- Test code and Documentation
- Clean the code
- More on LSTM basics/tutorials
