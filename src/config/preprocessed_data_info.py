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
                 train_files_path,
                 val_files_path,
                 test_path_files,
                 words_vocab_file,
                 chars_vocab_file,
                 entity_vocab_file,
                 char_2_id_map):

        self.char_2_id_map = char_2_id_map
        self.VOCAB_SIZE = vocab_size
        self.CHAR_VOCAB_SIZE = len(char_2_id_map)
        self.NUM_TAGS = num_tags

        self.TEXT_COL = text_col
        self.ENTITY_COL = entity_col
        self.ENTITY_IOB_COL = entity_iob_col

        self.TRAIN_FILES_PATH = os.path.abspath(train_files_path)
        self.VAL_FILES_PATH = os.path.abspath(val_files_path)
        self.TEST_FILES_PATH = os.path.abspath(test_path_files)

        self.WORDS_VOCAB_FILE = os.path.abspath(words_vocab_file)
        self.CHARS_VOCAB_FILE = os.path.abspath(chars_vocab_file)
        self.ENTITY_VOCAB_FILE = os.path.abspath(entity_vocab_file) #TODO make this as in memory data

    @staticmethod
    def is_file_exists(experiment_data_dir):
         return os.path.exists(experiment_data_dir + "config/processed_data_info.pickle")

    @staticmethod
    def save(info, experiment_data_dir):
        print_info("Storing the PreprocessedDataInfo for further use... \n{}\n ".format(info))

        if not os.path.exists(experiment_data_dir + "/config/processed_data_info.pickle"):
            with open(experiment_data_dir + "/config/processed_data_info.pickle", "wb") as file:
                pickle.dump(info, file=file)

    @staticmethod
    def load(experiment_data_dir):
        info = None
        try:
            print_info("Loading {}".format(experiment_data_dir + "/config/processed_data_info.pickle"))
            with open(experiment_data_dir + "/config/processed_data_info.pickle", "rb") as file:
                info = pickle.load(file)
            print_info("Restoring the PreprocessedDataInfo for further use... \n{}\n ".format(info))
        except:
            info = None
            print_info("{} is missing!!!".format(experiment_data_dir + "/config/processed_data_info.pickle"))

            raise EnvironmentError

        return info