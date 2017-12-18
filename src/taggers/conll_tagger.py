import sys

sys.path.append("src/")

from models.model_factory import TFEstimatorFactory
from helpers.print_helper import *

from preprocessor.conll_data_preprocessor import CoNLLDataPreprocessor
from data_iterators.conll_data_iterator import CoNLLDataIterator

EXPERIMENT_ROOT_DIR = "conll_experiments/"
# MODEL_DIR = "/opt/0.imaginea/git/i-tagger/conll_experiments/bilstm_crf_v0/charembd_True_lr_0.001_lstmsize_2-32-32_wemb_32_cemb_32_outprob_0.5"

NUM_EPOCHS = 15
BATCH_SIZE = 32


class CoNLLTagger():
    def __init__(self,model_dir=None):
        self.preprocessor = None
        self.estimator = None
        self.data_iterators = None
        self.model_dir=model_dir

        self.preprocessor = CoNLLDataPreprocessor(
            experiment_root_directory=EXPERIMENT_ROOT_DIR,
            over_write=None,
            use_iob=None,
            out_dir=None,
            train_csvs_path=None,
            val_csv_path=None,
            test_csv_path=None,
            db_reference_file=None,
            text_col=None,
            entity_col=None,
            do_run_time_config=False)

    def load_estimator(self):
        estimator_config, estimator = TFEstimatorFactory.get("bilstm_crf_v0")
        if self.model_dir:
            config = estimator_config.load(self.model_dir)
            if config == None:  # Fail safe
                estimator_config = estimator_config.with_user_hyperparamaters(EXPERIMENT_ROOT_DIR,
                                                                              self.preprocessor.OUT_DIR)
            else:
                estimator_config = config
        else:
            estimator_config = estimator_config.with_user_hyperparamaters(EXPERIMENT_ROOT_DIR,
                                                                          self.preprocessor.OUT_DIR)

        self.estimator = estimator(estimator_config)

        self.data_iterators = CoNLLDataIterator(self.preprocessor.OUT_DIR, batch_size=BATCH_SIZE)

    def preprocess(self):
        self.preprocessor.start()

    def train(self):
        self.load_estimator()

        if self.estimator.FEATURE_TYPE != self.data_iterators.FEATURE_TYPE:
            print_error("Given DataIterator can be used with choosed model. Try other models!!!")

        self.data_iterators.prepare()
        num_samples = self.data_iterators.NUM_TRAINING_SAMPLES

        print_info(num_samples)
        max_steps = (num_samples // BATCH_SIZE) * NUM_EPOCHS
        print_info("Total number steps: {} ".format(max_steps))

        for current_epoch in range(NUM_EPOCHS):
            max_steps = (num_samples // BATCH_SIZE) * (current_epoch + 1)

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

    def predict_on_test_files(self):
        self.data_iterators.predict_on_csv_files(estimator=self.estimator,
                                                 csv_files_path="data/test/")
