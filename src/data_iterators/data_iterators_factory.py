import sys
sys.path.append("../")
sys.path.append("src/")

import tensorflow as tf
import argparse
from importlib import import_module

class DataIteratorsFactory():

    data_iterators = {
        "csv_data_iterator": "CsvDataIterator",
    }

    def __init__(self):
        ""

    @staticmethod
    def _get_preprocessor(name):
        '''
        '''
        try:
            preprocessor = getattr(import_module("data_iterators." + name), DataIteratorsFactory.data_iterators[name])
        except KeyError:
            raise NotImplemented("Given data_iterators file name not found: {}".format(name))
        return preprocessor


    @staticmethod
    def get(preprocessor_file_name="csv_data_iterator"):
        preprocessor = DataIteratorsFactory._get_preprocessor(preprocessor_file_name)
        return preprocessor


