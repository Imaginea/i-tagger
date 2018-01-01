import sys
sys.path.append("../")
import os
from helpers.os_helper import *
from config.config_helper import ConfigManager

class IPreprocessorInterface():

    def __init__(self, experiment_root_directory):
        self.EXPERIMENT_ROOT_DIR = experiment_root_directory
        self.OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + "processed_data"
        self.load_ini()

    def load_ini(self):
        '''
        Assuming each dataset will have its own configuration, a `experiment_folder/config/*.ini`
        is used to store and read data specific configuration
        :return: 
        '''
        self.config = ConfigManager(self. EXPERIMENT_ROOT_DIR + "/config/preprocessor.ini")

    def create_target_directories(self):
        '''
        We wish not to preprocess the data every time, so before storing the preprocessed
        data appropriate data folders are created and **over write** should be handled
        :return: 
        '''
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def extract_vocab(self):
        raise NotImplementedError

    def save_preprocessed_data_info(self):
        raise NotImplementedError

    def preprocess(self):
        self.create_target_directories()
        self.prepare_data()
        self.extract_vocab()
        self.save_preprocessed_data_info()

