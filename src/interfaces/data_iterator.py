import sys
sys.path.append("../")
import tensorflow as tf

class DataIterator():
    def __init__(self, model_fn, model_dir, config):
       ""

    def load_dats_config(self):
        raise NotImplementedError

    def make_seq_pair(self):
        raise NotImplementedError

    def setup_input_graph(self):
        raise NotImplementedError
