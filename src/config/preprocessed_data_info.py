import sys
import os
import pickle
sys.path.append("../")
from helpers.print_helper import *

class PreprocessedDataInfo():
    def __init__(self,
                 vocab_size,
                 num_tags,
                 text_col,
                 entity_col,
                 entity_iob_col,
                 train_data_file,
                 val_data_file,
                 test_data_file,
                 words_vocab_file,
                 chars_vocab_file,
                 entity_vocab_file,
                 char_2_id_map):
        self.VOCAB_SIZE = vocab_size
        self.NUM_TAGS = num_tags

        self.TEXT_COL = text_col
        self.ENTITY_COL = entity_col
        self.ENTITY_IOB_COL = entity_iob_col

        self.TRAIN_DATA_FILE = os.path.abspath(train_data_file)
        self.VAL_DATA_FILE = os.path.abspath(val_data_file)
        self.TEST_DATA_FILE = os.path.abspath(test_data_file)

        self.WORDS_VOCAB_FILE = os.path.abspath(words_vocab_file)
        self.CHARS_VOCAB_FILE = os.path.abspath(chars_vocab_file)
        self.ENTITY_VOCAB_FILE = os.path.abspath(entity_vocab_file)

        self.char_2_id_map = char_2_id_map

    @staticmethod
    def save(info, data_dir):
        print_info("Storing the PreprocessedDataInfo for further use... \n{}\n ".format(info))

        with open(data_dir + "/processed_data_info.pickle", "wb") as file:
            pickle.dump(info, file=file)

    @staticmethod
    def load(data_dir):
        with open(data_dir + "/processed_data_info.pickle", "wb") as file:
            info = pickle.load(file)
        print_info("Restoring the PreprocessedDataInfo for further use... \n{}\n ".format(info))

        return info