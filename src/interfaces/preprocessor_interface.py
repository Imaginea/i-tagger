import sys
sys.path.append("../")
import os
from helpers.os_helper import *

class IPreprocessorInterface():

    def __init__(self, experiment_root_directory):
        self.load_ini()
        self. EXPERIMENT_ROOT_DIR = experiment_root_directory

    def load_ini(self):
        '''
        Assuming each dataset will have its own configuration, a `src/config/*.ini`
        is used to store and read data specific configuration
        :return: 
        '''
        raise NotImplementedError

    def create_target_directories(self):
        '''
        We wish not to preprocess the data every time, so before storing the preprocessed
        data appropriate data folders are created and **over write** should be handled dynamically
        :return: 
        '''
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def extract_vocab(self):
        raise NotImplementedError

    def save_preprocessed_data_info(self):
        raise NotImplementedError

    def start(self):
        self.create_target_directories()
        self.prepare_data()
        self.extract_vocab()
        self.save_preprocessed_data_info()

