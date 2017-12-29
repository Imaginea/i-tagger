## How to test?

### Commands
- For running patent dataset example move to directory: i-tagger

#### CoNLL

We are using the CoNLL 2003 dataset for testing.


**Data :**
- https://www.clips.uantwerpen.be/conll2003/ner/
- https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003

```bash
cd /path/to/i-tagger/

python src/commands/conll_2003_dataset.py --mode=preprocess

#go with default settings

python src/commands/conll_2003_dataset.py --mode=train \
--model-name=bilstm_crf_v0 \
--batch-size=64 \
--num-epochs=5

PATH_TO_MODEL=conll_experiments/bilstm_crf_v0/charembd_True_lr_0.001_lstmsize_2-48-32_wemb_48_cemb_32_outprob_0.5/

#Note: while retraining kep the batch size consistent

python src/commands/conll_2003_dataset.py --mode=retrain \
--model-dir=$PATH_TO_Model \
--batch-size=64 \
--num-epochs=5

python src/commands/conll_2003_dataset.py --mode=predict --model-dir=$PATH_TO_Model --predict-dir=PATH to Prediction files
```

```
tensorboard --logdir=$PATH_TO_MODEL
```


**Demo on web interface**

A pretrained model on CoNLL data set is available on `path/to/i-tagger/conll_experiments/bilstm_crf_v0/charembd_True_lr_0.001_lstmsize_2-48-32_wemb_48_cemb_32_outprob_0.5`

Fire up the web app and test it ;)

```bash
python src/app.py
```
- Open browser and navigate to : http://localhost:8080
  Here user can select an option either to get the tags by entering the text or using file upload.

- Open browser and navigate to : http://localhost:8080/predictText
  Here user can enter any text to get the tags for the entered text.

Eg: He was well backed by England hopeful Mark Butcher who made 70 as Surrey closed on 429 for seven, a lead of 234.
OR

- Open browser and navigate to : http://localhost:8080/predict
  Here user can upload a file and get the tags for the uploaded file.

![](../docs/images/web_app.png)