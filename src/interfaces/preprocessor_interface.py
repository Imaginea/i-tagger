import sys
sys.path.append("../")
import os
from helpers.os_helper import *

class IPreprocessorInterface():

    def __init__(self, experiment_root_directory="experiments"):
        self.load_ini()
        self. EXPERIMENT_ROOT_DIR = experiment_root_directory

    def load_ini(self):
        raise NotImplementedError

    def create_target_directories(self):
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

