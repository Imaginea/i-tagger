import sys
import tensorflow as tf
import argparse

sys.path.append("src/")

from taggers.conll_tagger import CoNLLTagger

def get_tf_flags():

    flags = tf.app.flags

    flags.DEFINE_string("action","none","preprocess/train/retrain")

    flags.DEFINE_string("data_dir","experiments/tf_data/","")


    cfg = tf.app.flags.FLAGS
    return cfg

def run(opt):
    tagger = CoNLLTagger(model_dir=opt.model_dir)
    if opt.mode =="preprocess":
        tagger.preprocess()
    elif opt.mode == "train":
        tagger.train()
    elif opt.mode =="retrain":
        tagger.train()




if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Prepare data for Tensorflow training...")

    #CONLL specific preprocessing

    optparse.add_argument('-mode','--mode',
                          choices=['preprocess', 'train', "retrain"],
                          required=True,
                          help="'preprocess, 'train', 'retrain'"
                          )

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir', required=False,
                          help='Pass to model directory needed for training')





    opt = optparse.parse_args()
    if opt.mode == 'retrain' and not opt.model_dir:
        optparse.error('-t argument is required in "retrain" mode.')
    else:
        run(opt)




