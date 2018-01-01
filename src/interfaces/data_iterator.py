import sys
sys.path.append("../")
import tensorflow as tf
from config.preprocessed_data_info import PreprocessedDataInfo
from helpers.print_helper import *


class IDataIterator():
    def __init__(self, experiment_dir, batch_size):
        '''
        Data Iterators with different features type are expected to 
        implement this interface, exposing the input functions and their hooks
        :param experiment_dir: 
        :param batch_size: 
        
        '''

        self.preprocessed_data_info = PreprocessedDataInfo.load(experiment_dir)

        self.BATCH_SIZE = batch_size
        self._train_data_input_fn = None
        self._train_data_init_hook = None

        self._val_data_input_fn= None
        self._val_data_init_hook = None

        self._test_data_input_fn= None
        self._test_data_init_hook = None

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

    # def prepare(self):
    #     self.setup_train_input_graph()
    #     self.setup_val_input_graph()
    #     self.setup_predict_input_graph()