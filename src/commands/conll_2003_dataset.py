import argparse
import sys

sys.path.append("src/")

from taggers.conll_tagger import CoNLLTagger


def run(opt):
    tagger = CoNLLTagger(model_dir=opt.model_dir)
    if opt.mode == "preprocess":
        tagger.preprocess()
    elif opt.mode == "train":
        tagger.train()
    elif opt.mode == "retrain":
        tagger.train()
    elif opt.mode == "predict":
        tagger.predict_on_test_files(abs_fpath=opt.predict_dir)


if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Prepare data for Tensorflow training...")

    # CONLL specific preprocessing

    optparse.add_argument('-mode', '--mode',
                          choices=['preprocess', 'train', "retrain", "predict"],
                          required=True,
                          help="'preprocess, 'train', 'retrain'"
                          )

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir', required=False,
                          help='Pass to model directory needed for training')

    optparse.add_argument('-pd', '--predict-dir', action='store',
                          dest='predict_dir', required=False,
                          help='Model directory needed for prediction')

    opt = optparse.parse_args()
    if (opt.mode == 'retrain' or opt.mode == 'predict') and not opt.model_dir:
        optparse.error('--model-dir argument is required in "retrain" & "predict" mode.')

    elif opt.mode == 'predict' and not opt.predict_dir:
        optparse.error('--predict-dir argument is required in "predict" mode.')
    else:
        run(opt)
