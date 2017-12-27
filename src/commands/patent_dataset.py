import argparse
import sys

sys.path.append("src/")

from taggers.patent_tagger import PatentTagger


def run(opt):
    tagger = PatentTagger(model_dir=opt.model_dir)
    if opt.mode == "preprocess":
        tagger.preprocess()
    elif opt.mode == "train":
        tagger.train(opt.batch_size, opt.num_epochs)
    elif opt.mode == "retrain":
        tagger.train(opt.batch_size, opt.num_epochs)
    elif opt.mode == "predict":
        tagger.predict_on_test_files(csv_files_path=opt.predict_dir)


if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Prepare data for Tensorflow training...")

    # CONLL specific preprocessing

    optparse.add_argument('-mode', '--mode',
                          choices=['preprocess', 'train', "retrain","predict"],
                          required=True,
                          help="'preprocess, 'train', 'retrain'"
                          )

    optparse.add_argument('-mn', '--model-name', action='store',
                          dest='model_name', required=False,
                          help='Name of the model python file to be used')

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir', required=False,
                          help='Pass to model directory needed for training')

    optparse.add_argument('-pd', '--predict-dir', action='store',
                          dest='predict_dir', required=False,
                          help='Model directory needed for prediction')


    optparse.add_argument('-bs', '--batch-size',  type=int, action='store',
                          dest='batch_size', required=False,
                          help='Batch size for training, be consistent when retraining')

    optparse.add_argument('-ne', '--num-epochs', type=int, action='store',
                          dest='num_epochs', required=False,
                          help='Number of epochs')

    opt = optparse.parse_args()
    if (opt.mode == 'retrain' or opt.mode == 'predict') and not opt.model_dir:
        optparse.error('--model-dir argument is required in "retrain" & "predict" mode.')

    elif opt.mode == 'predict' and not opt.predict_dir:
        optparse.error('--predict-dir argument is required in "predict" mode.')
    else:
        run(opt)