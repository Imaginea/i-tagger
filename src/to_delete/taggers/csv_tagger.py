import sys

sys.path.append("src/")

from models.model_factory import TFEstimatorFactory
from helpers.print_helper import *

from preprocessor.conll_data_preprocessor import CoNLLDataPreprocessor
from data_iterators.csv_data_iterator import CsvDataIterator

EXPERIMENT_ROOT_DIR = "conll_csv_experiments"

NUM_EPOCHS = 5
BATCH_SIZE = 2


class CsvTagger():
    def __init__(self, model_name="bilstm_crf_v0", model_dir=None):
        self.preprocessor = None
        self.estimator = None
        self.data_iterators = None
        self.model_dir = model_dir

        self.model_name = model_name

        #fail safe
        if self.model_name == None:
            self.model_name = "bilstm_crf_v0"

        self.preprocessor = CoNLLDataPreprocessor(
            experiment_root_directory=EXPERIMENT_ROOT_DIR,
            over_write=None,
            use_iob=None,
            out_dir=None,
            train_df_path=None,
            val_df_path=None,
            test_df_path=None,
            text_col=None,
            entity_col=None,
            do_run_time_config=False)

    def load_estimator(self):
        estimator_config, estimator = TFEstimatorFactory.get(self.model_name)
        if self.model_dir:
            config = estimator_config.load(self.model_dir)
            if config is None:  # Fail safe
                estimator_config = estimator_config.with_user_hyperparamaters(EXPERIMENT_ROOT_DIR,
                                                                              self.preprocessor.DATA_OUT_DIR)
            else:
                estimator_config = config
        else:
            estimator_config = estimator_config.with_user_hyperparamaters(EXPERIMENT_ROOT_DIR,
                                                                          self.preprocessor.DATA_OUT_DIR)
        self.estimator = estimator(estimator_config)
        self.data_iterators = CsvDataIterator(self.preprocessor.DATA_OUT_DIR, batch_size=BATCH_SIZE)

    def preprocess(self):
        self.preprocessor.start()

    def train(self, batch_size, num_epochs):
        self.load_estimator()

        if self.estimator.FEATURE_NAME != self.data_iterators.FEATURE_NAME:
            print_error("Given DataIterator can be used with choosed model. Try other models!!!")
            exit(1)

        self.data_iterators.prepare()
        num_samples = self.data_iterators.NUM_TRAINING_SAMPLES

        print_info(num_samples)
        max_steps = (num_samples // batch_size) * num_epochs
        print_info("Total number steps: {} ".format(max_steps))

        for current_epoch in range(num_epochs):
            max_steps = (num_samples // batch_size) * (current_epoch + 1)

            train_hooks = []
            train_hooks.append(self.data_iterators.train_data_init_hook)
            # if len(estimator.hooks) > 0:
            #     train_hooks.extend(tagger.hooks)

            self.estimator.train(input_fn=self.data_iterators.train_data_input_fn,
                     hooks=train_hooks,
                     max_steps=max_steps)

            eval_results = self.estimator.evaluate(input_fn=self.data_iterators.val_data_input_fn,
                                           hooks=[self.data_iterators.val_data_init_hook])  # tf_debug.LocalCLIDebugHook()

            print(eval_results)


    def predict_on_test_files(self, csv_files_path="data/test/"):
        # TODO handle the estimator behaviour in train, retrain and predict mode
        if self.estimator.FEATURE_NAME == self.data_iterators.FEATURE_NAME:
            self.load_estimator()
            self.data_iterators.predict_on_csv_files(estimator=self.estimator, csv_files_path=csv_files_path)
        else:
            print_error("Given DataIterator can be used with choosed model. Try other models!!!")
            exit(1)