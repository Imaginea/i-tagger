import sys
sys.path.append("../")
import tensorflow as tf
from config.preprocessed_data_info import PreprocessedDataInfo
from helpers.print_helper import *


class IDataIterator():
    def __init__(self, data_dir, batch_size):
       self.preprocessed_data_info = PreprocessedDataInfo.load(data_dir)

       self.BATCH_SIZE = batch_size
       self.train_data_input_fn, self.train_data_init_hook = None, None

       self.val_data_input_fn= None
       self.val_data_init_hook = None

       self.test_data_input_fn= None
       self.test_data_init_hook = None

    def setup_train_input_graph(self):
        raise NotImplementedError

    def setup_val_input_graph(self):
        raise NotImplementedError

    def setup_predict_input_graph(self):
        print_warn("No user implementation for predictions")

    def prepare(self):
        self.setup_train_input_graph()
        self.setup_val_input_graph()
        self.setup_predict_input_graph()