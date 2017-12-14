import sys
import tensorflow as tf

sys.path.append("src/")

from taggers.conll_tagger import CoNLLTagger

def get_tf_flags():

    flags = tf.app.flags

    flags.DEFINE_string("action","none","preprocess/train/retrain")

    flags.DEFINE_string("data_dir","experiments/tf_data/","")


    cfg = tf.app.flags.FLAGS
    return cfg


tagger = CoNLLTagger()
tagger.preprocess()
tagger.train()





