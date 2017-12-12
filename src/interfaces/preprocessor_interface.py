import sys
sys.path.append("../")
import os
from helpers.os_helper import *

class PreprocessorInterface():

    def __init__(self):
        self.load_ini()

    def load_ini(self):
        raise NotImplementedError

    def create_target_directories(self):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def extract_vocab(self):
        raise NotImplementedError

    def dumb_config(self):
        raise NotImplementedError

    def start(self):
        self.create_target_directories()
        self.prepare_data()
        self.extract_vocab()
        self.dumb_config()

