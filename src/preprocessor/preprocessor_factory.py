import sys
sys.path.append("../")
sys.path.append("src/")

import tensorflow as tf
import argparse
from importlib import import_module

class PreprocessorFactory():

    preprocessors = {
        "conll_data_preprocessor": "CoNLLDataPreprocessor",
        "csv_data_preprocessor": "CoNLLDataPreprocessor",
        "patent_data_preprocessor" : "PatentDataPreprocessor"
    }

    def __init__(self):
        ""

    @staticmethod
    def _get_preprocessor(name):
        '''
        '''
        try:
            preprocessor = getattr(import_module("preprocessor." + name), PreprocessorFactory.preprocessors[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        return preprocessor


    @staticmethod
    def get(model_name="csv_data_preprocessor"):
        preprocessor = PreprocessorFactory._get_preprocessor(model_name)
        return preprocessor


