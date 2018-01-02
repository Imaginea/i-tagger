import argparse
import sys
sys.path.append("src/")

from data_iterators.data_iterators_factory import DataIteratorsFactory
from helpers.print_helper import *

from models.model_factory import TFEstimatorFactory

from preprocessor.preprocessor_factory import PreprocessorFactory
from post_processor.naive_metrics import get_naive_metrics
from tqdm import tqdm

def load_estimator(experiment_name, data_iterator, model_name, model_dir):
    # Use the factory loader to load the appropriate model
    estimator_config, estimator = TFEstimatorFactory.get(model_name)

    # If model-dir is passed by the user, we can safely assume
    # that there exists a model directory and load the saved config
    if model_dir:
        config = estimator_config.load(model_dir)
        if config is None:  # Fail safe
            estimator_config = estimator_config.with_user_hyperparamaters(experiment_name, data_iterator)
        else:
            estimator_config = config
    else:
        # Each estimator is expected to take user inputs for model hyper parameters
        # and create a directory like experiment_folder/model_name/hyper_parameters/
        # Also it stores the model config file as a pickle in experiment_folder/model_name/hyper_parameters/
        estimator_config = estimator_config.with_user_hyperparamaters(experiment_name, data_iterator)

    estimator = estimator(estimator_config)

    return estimator


def train(estimator, data_iterator, batch_size, num_epochs):

    # It is expected that both estimator and data iterators are working on
    # same set of features, which is enforced here by using  Feature types
    if estimator.FEATURE_NAME != data_iterator.FEATURE_NAME:
        print_error("Given DataIterator can be used with choosed model. Try other models!!!")
        exit(1)

    train_hooks = []
    train_hooks.append(data_iterator.train_data_init_hook)
    #TODO currently NUM_TRAINING_SAMPLES initialized on first acces on input fn/hook

    # Info needed for calculation
    num_samples = data_iterator.NUM_TRAINING_SAMPLES

    print_info(num_samples)
    max_steps = (num_samples // batch_size) * num_epochs
    print_info("Total number steps: {} ".format(max_steps))

    for current_epoch in tqdm(range(num_epochs)):
        max_steps = (num_samples // batch_size) * (current_epoch + 1)

        # if len(estimator.hooks) > 0:
        #     train_hooks.extend(tagger.hooks)

        estimator.train(input_fn=data_iterator.train_data_input_fn,
                             hooks=train_hooks,
                             max_steps=max_steps)

        eval_results = estimator.evaluate(input_fn=data_iterator.val_data_input_fn,
                                               hooks=[data_iterator.val_data_init_hook])  # tf_debug.LocalCLIDebugHook()

        print(eval_results)


def run(opt):
    # Use the factory loader to load the appropriate pre-processor,
    # which depends/reads the config from experiment_folder/config/*.ini
    if opt.mode == "preprocess":
        preprocessor = PreprocessorFactory.get(opt.preprocessor_name)
        # Initialize the preprocessor with experiment folder path
        preprocessor = preprocessor(opt.experiment_name)
        preprocessor.preprocess()
        exit(0)

    # Use the factory loader to load the appropriate data-iterator,
    # which depends/reads the config from experiment_folder/config/*.pickle
    # stored previously by preprocessor
    data_iterator = DataIteratorsFactory.get(opt.data_iterator_name)
    # Initialize the data iterator with experiment folder path and batch size
    # all other needed config/info are read from the *.pickle file
    data_iterator = data_iterator(opt.experiment_name, opt.batch_size)
    # data_iterator.prepare()

    # Load the estimator
    estimator = load_estimator(opt.experiment_name, data_iterator, opt.model_name, opt.model_dir)

    if opt.mode == "train":
        train(estimator, data_iterator, opt.batch_size, opt.num_epochs)
    elif opt.mode == "retrain":
        train(estimator, data_iterator, opt.batch_size, opt.num_epochs)
    elif opt.mode == "predict":
        # Each data iterator has it own way of handling features,
        # hence pass the estimator and the files to be tagged
        # out_dir = data_iterator.predict_on_test_files(estimator, csv_files_path=opt.predict_dir)

        print_info("Calculating some naive metrics...")
        get_naive_metrics(predicted_csvs_path=estimator.model_dir + "/predictions/",
                          ner_tag_vocab_file=data_iterator.ENTITY_VOCAB_FILE,
                          entity_col_name=data_iterator.ENTITY_COL,
                          prediction_col_name="predictions",
                          out_dir="/opt/0.imaginea/git/i-tagger/conll_csv_experiments/csv_data_iterator/bilstm_crf_v0/charembd_True_lr_0.001_lstmsize_2-64-48_wemb_64_cemb_48_outprob_0.5/")

if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Sequence modeling with Tensorflow ...")

    optparse.add_argument('-en', '--experiment-name', action='store',
                          dest='experiment_name', required=False,
                          help='Folder where data, config, models and output resides')

    optparse.add_argument('-mode', '--mode',
                          choices=['preprocess', 'train', "retrain","predict"],
                          required=True,
                          help="'preprocess, 'train', 'retrain', 'predict'"
                          )

    optparse.add_argument('-pn', '--preprocessor-name', action='store',
                          dest='preprocessor_name', required=False,
                          help='Name of the preprocessor python file to be used')

    optparse.add_argument('-di', '--data-iterator-name', action='store',
                          dest='data_iterator_name', required=False,
                          help='Name of the data-iterator python file to be used')

    optparse.add_argument('-mn', '--model-name', action='store',
                          dest='model_name', required=False,
                          help='Name of the model python file to be used')

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir', required=False,
                          help='Pass to model directory needed for re-training')

    optparse.add_argument('-pd', '--predict-dir', action='store',
                          dest='predict_dir', required=False,
                          help='Model directory needed for prediction')

    optparse.add_argument('-bs', '--batch-size',  type=int, action='store',
                          dest='batch_size', required=False,
                          help='Batch size for training,  and should be consistent when retraining')

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