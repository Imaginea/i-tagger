import sys
sys.path.append("../")
sys.path.append("src/")

from importlib import import_module

class DataIteratorsFactory():

    data_iterators = {
        "csv_data_iterator": "CsvDataIterator",
        "positional_patent_data_iterator": "PositionalPatentDataIterator",
        "csv_pos_data_iterator": "CsvPOSDataIterator"
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


