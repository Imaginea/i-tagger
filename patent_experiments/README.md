## How to test?

### Commands
- For running patent dataset example move to directory: i-tagger


**About Data :**
- This part of the module takes into account auxillary features present in the dataset.
- In this case, every word is having its postional features (X,Y,PageNo), where X is the x cordinate of the word,Y is the y cordinate of the word and PageNo defines the page no from which the word is taken.
- Although this is the case currently, we can replace these features with any numeric features or on hot-encoded text-features. These features are merged with the output of the BiLSTM layer eventually.



```bash
cd /path/to/i-tagger/


# Below code only transforms the data

python src/commands/tagger.py \
--experiment-name=patent_experiments \
--mode=preprocess \
--preprocessor-name=patent_data_preprocessor

# 1. Data iterator is were main core logic of data preparation for the models
# happens, which should implement IDataIterator and inherit an Feature type
# 2. Model takes in the data iterator while configuring and reads
# required information from it along with user params and initializes the
# model
# 3. Tensorflow estimator is used then to train with above
# model and data iterators

python src/commands/tagger.py \
--experiment-name=patent_experiments \
--mode=train \
--data-iterator-name=positional_patent_data_iterator \
--model-name=bilstm_crf_v1 \
--batch-size=5 \
--num-epochs=5

#above when run with default params will create following model
export MODEL_PATH=patent_experiments/positional_patent_data_iterator/bilstm_crf_v1/charembd_True_lr_0.001_lstmsize_2-48-32_wemb_48_cemb_32_outprob_0.5/

python src/commands/tagger.py \
--experiment-name=patent_experiments \
--mode=retrain \
--data-iterator-name=positional_patent_data_iterator \
--model-name=bilstm_crf_v1 \
--batch-size=5 \
--num-epochs=8 \
--model-dir=$MODEL_PATH

python src/commands/tagger.py \
--experiment-name=patent_experiments \
--mode=predict \
--data-iterator-name=positional_patent_data_iterator \
--model-name=bilstm_crf_v1 \
--model-dir=$MODEL_PATH \
--predict-dir=patent_experiments/preprocessed_data/test/
```


```
tensorboard --logdir=$MODEL_PATH
```

**Training Without Positional Features :**
- If you decide to train the patent dataset without positional features, this can be done easily as defined below.
- The components are designed in such a way that they can be used in combination with each other without having to re-write code.
- For training the patent data without positional features, we will use the patent data preprocessor and csv data iterator.
- The model will be pointed to the  bilstm_crf_v0 (Model without positional features in the network)


```
python src/commands/tagger.py \
--experiment-name=patent_non_positional_experiments \
--mode=preprocess \
--preprocessor-name=patent_data_preprocessor

python src/commands/tagger.py \
--experiment-name=patent_non_positional_experiments \
--mode=train \
--data-iterator-name=csv_data_iterator \
--model-name=bilstm_crf_v0 \
--batch-size=3 \
--num-epochs=5

#above when run with default params will create following model
export MODEL_PATH=patent_non_positional_experiments/csv_data_iterator/bilstm_crf_v0/charembd_True_lr_0.001_lstmsize_2-64-48_wemb_64_cemb_48_outprob_0.5

python src/commands/tagger.py \
--experiment-name=patent_non_positional_experiments \
--mode=retrain \
--data-iterator-name=csv_data_iterator \
--model-name=bilstm_crf_v0 \
--batch-size=3 \
--num-epochs=5 \
--model-dir=$MODEL_PATH

python src/commands/tagger.py \ 
--experiment-name=patent_non_positional_experiments --mode=predict \
--data-iterator-name=csv_data_iterator --model-name=bilstm_crf_v0 \
--model-dir=$MODEL_PATH \
--predict-dir=patent_non_positional_experiments/preprocessed_data/test/

```
