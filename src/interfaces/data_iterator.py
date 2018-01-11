import sys
sys.path.append("../")
from helpers.print_helper import *
from config.config_helper import ConfigManager
from helpers.os_helper import check_n_makedirs

class IDataIterator():
    def __init__(self, name, experiment_dir, batch_size):
        '''
        Data Iterators with different features type are expected to 
        implement this interface, exposing the input functions and their hooks
        :param experiment_dir: 
        :param batch_size: 
        
        '''

        self.NAME = name
        self.EXPERIMENT_ROOT_DIR = experiment_dir
        self.OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + self.NAME + "/"

        self._load_ini()
        # self.preprocessed_data_info = PreprocessedDataInfo.load(experiment_dir)

        # This rule is assumed to be correct if the previous stage is of IPreprocessorInterface
        self.PREPROCESSED_DATA_DIR = self.EXPERIMENT_ROOT_DIR + "/" + self.config.get_item("OutputDirectories","preprocessed_data_dir")
        self.TRAIN_FILES_IN_PATH = self.PREPROCESSED_DATA_DIR + "/train/"
        self.VAL_FILES_IN_PATH = self.PREPROCESSED_DATA_DIR + "/val/"
        self.TEST_FILES_IN_PATH = self.PREPROCESSED_DATA_DIR + "/test/"

        self.TEXT_COL = self.config.get_item("Schema", "text_column")
        self.ENTITY_COL = self.config.get_item("Schema", "entity_column")
        self.WORDS_VOCAB_FILE = self.OUT_DIR + "/" + self.TEXT_COL + "_" + "vocab.tsv"
        self.CHARS_VOCAB_FILE = self.OUT_DIR + "/" + self.TEXT_COL + "_" + "chars_vocab.tsv"
        self.ENTITY_VOCAB_FILE = self.OUT_DIR + "/" + self.ENTITY_COL + "_vocab.tsv"

        check_n_makedirs(self.OUT_DIR)

        self.BATCH_SIZE = batch_size
        self.NUM_TAGS = None
        self.VOCAB_SIZE = None
        self.CHAR_VOCAB_SIZE = None

        self._train_data_input_fn = None
        self._train_data_init_hook = None

        self._val_data_input_fn= None
        self._val_data_init_hook = None

        self._test_data_input_fn= None
        self._test_data_init_hook = None

    def _load_ini(self):
        '''
        Assuming each dataset will have its own configuration, a `experiment_folder/config/*.ini`
        is used to store and read data specific configuration
        :return: 
        '''
        self.config = ConfigManager(self.EXPERIMENT_ROOT_DIR + "/config/config.ini")

    def setup_train_input_graph(self):
        raise NotImplementedError

    def setup_val_input_graph(self):
        raise NotImplementedError

    def setup_test_input_graph(self):
        print_warn("No user implementation for predictions")

    @property
    def train_data_input_fn(self):
        if self._train_data_input_fn is None:
            self.setup_train_input_graph()
        return self._train_data_input_fn

    @property
    def train_data_init_hook(self):
        if self._train_data_init_hook is None:
            self.setup_train_input_graph()
        return self._train_data_init_hook

    @property
    def val_data_input_fn(self):
        if self._val_data_input_fn is None:
            self.setup_val_input_graph()
        return self._val_data_input_fn

    @property
    def val_data_init_hook(self):
        if self._val_data_init_hook is None:
            self.setup_val_input_graph()
        return self._val_data_init_hook

    @property
    def test_data_input_fn(self):
        if self._test_data_input_fn is None:
            self.setup_test_input_graph()
        return self._test_data_input_fn

    @property
    def test_data_init_hook(self):
        if self._test_data_init_hook is None:
            self.setup_test_input_graph()
        return self._test_data_init_hook

    def predict_on_dataframes(self, estimator, dfs):
        '''
        Implement this function to predict on given data frame
        based on configured input/text column
        :param estimator: One of the models that support this data iterator
        :param dfs: Pandas data frames of the test/user file/s
        :return: New data frame with predicted columns
        '''
        raise NotImplementedError

    def predict_on_test_files(self, estimator, df_files_path):
        '''
        Iterate through the files and use `predict_on_test_file`, for prediction
        :param estimator: One of the models that support this data iterator
        :param df_files_path: Files that can be opened by the pandas 
        :return: Creates a folder estimator.model_dir/predictions/ and adds the predicted files
        '''
        raise NotImplementedError

    def predict_on_text(self, estimator, sentence):
        '''
        Use this for user interaction on the fly
        :param estimator: One of the models that support this data iterator
        :param sentence: Text deliminated by space
        :return: 
        '''
        raise NotImplementedError