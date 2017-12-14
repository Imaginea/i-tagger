import sys
sys.path.append("../")
import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from tensorflow.python.platform import gfile

from interfaces.preprocessor_interface import IPreprocessorInterface
from helpers.print_helper import *
from config.global_constants import *
from helpers.os_helper import check_n_makedirs
from config.config_helper import ConfigManager

from nlp.spacy_helper import naive_vocab_creater, get_char_vocab, vocab_to_tsv
from config.preprocessed_data_info import PreprocessedDataInfo

class CoNLLDataPreprocessor(IPreprocessorInterface):
    def __init__(self,
                 experiment_root_directory="experiments",
                 over_write=None,
                 use_iob=None,
                 out_dir=None,
                 train_csvs_path=None,
                 val_csv_path=None,
                 test_csv_path=None,
                 db_reference_file=None,
                 text_col=None,
                 entity_col=None,
                 do_run_time_config=False):
        '''
        
        :param over_write: 
        :param use_iob: 
        :param out_dir: 
        :param train_csvs_path: 
        :param val_csv_path: 
        :param test_csv_path: 
        :param db_reference_file: 
        :param text_col: 
        :param entity_col: 
        :param do_run_time_config: Enable this to use constructor params or 
        by default it uses config/patent_data_preprocessor.ini config
        '''
        super(CoNLLDataPreprocessor, self).__init__(experiment_root_directory)

        if do_run_time_config:
            self.OVER_WRITE = over_write
            self.USE_IOB = use_iob

            self.OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + out_dir

            self.TRAIN_CSV_PATH = train_csvs_path
            self.VAL_CSV_PATH = val_csv_path
            self.TEST_CSV_PATH = test_csv_path
            self.DB_REFERENCE_FILE = db_reference_file

            self.TEXT_COL = text_col
            self.ENTITY_COL = entity_col
            self.ENTITY_IOB_COL = entity_col + "_iob"
        else:
            self.OVER_WRITE = self.config.get_item("Options", "over_write")

            self.OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + self.config.get_item("OutputDirectories", "data_dir")

            self.TRAIN_DATA_FILE = self.config.get_item("InputFiles", "train_file_path")
            self.VAL_DATA_FILE = self.config.get_item("InputFiles", "val_file_path")
            self.TEST_DATA_FILE = self.config.get_item("InputFiles", "test_file_path")

            self.TEXT_COL = self.config.get_item("Schema", "text_column")
            self.ENTITY_COL = self.config.get_item("Schema", "entity_column")
            self.EXTRA_COLS = self.config.get_item("Schema", "extra_columns")
            self.EXTRA_COLS = [str.strip(col) for col in self.EXTRA_COLS.split(",")]

        self.WORDS_VOCAB_FILE = self.OUT_DIR + "/" + self.TEXT_COL + "_" + "vocab.tsv"
        self.CHARS_VOCAB_FILE = self.OUT_DIR + "/" + self.TEXT_COL + "_" + "chars_vocab.tsv"
        self.ENTITY_VOCAB_FILE = self.OUT_DIR + "/" + self.ENTITY_COL + "_vocab.tsv"

        self.COLUMNS = [self.TEXT_COL, self.ENTITY_COL]
        self.COLUMNS.extend(self.EXTRA_COLS)


    def load_ini(self):
        self.config = ConfigManager("src/config/conll_data_preprocessor.ini")

    def create_target_directories(self):
        if os.path.exists(self.OUT_DIR):
            if self.OVER_WRITE == "yes":
                print_info("Deletingls data folder: {}".format(self.OUT_DIR))
                shutil.rmtree(self.OUT_DIR)
                print_info("Recreating data folder: {}".format(self.OUT_DIR))
                os.makedirs(self.OUT_DIR)
            else:
                print_info("Skipping preprocessing step, since the data is already available")
                return "skip"
        else:
            print_info("Creating data folder: {}".format(self.OUT_DIR))
            os.makedirs(self.OUT_DIR)

    def prepare_data(self):
        print_info("Preparing train data...")

    def extract_vocab(self):
        if not os.path.exists(self.WORDS_VOCAB_FILE) and not os.path.exists(self.ENTITY_VOCAB_FILE):
            print_info("Preparing the vocab for the text col: {}".format(self.TEXT_COL))
            # Read the text file as DataFrame and extract vocab for text column and entity column
            train_df = pd.read_csv(self.TRAIN_DATA_FILE, sep=SEPRATOR, header=None)

            train_df.columns = self.COLUMNS  # just for operation, names doesn't imply anything here
            train_df.head()

            # Get word level vocab
            lines = train_df[self.TEXT_COL].astype(str).unique().tolist()
            # VOCAB_SIZE, words_vocab = tf_vocab_processor(lines, WORDS_VOCAB_FILE)
            self.VOCAB_SIZE, words_vocab = naive_vocab_creater(lines, self.WORDS_VOCAB_FILE, vocab_filter=True)

            # Get char level vocab
            words_chars_vocab = ['<P>', '<U>']
            _vocab = get_char_vocab(words_vocab)
            words_chars_vocab.extend(_vocab)

            # Create char2id map
            vocab_to_tsv(words_chars_vocab, self.CHARS_VOCAB_FILE)
            self.char_2_id_map = {c: i for i, c in enumerate(words_chars_vocab)}

            print_info("Preparing the vocab for the entity col: {}".format(self.ENTITY_COL))

            # Reopen the file without filling UNKNOWN_WORD in blank lines
            train_df = pd.read_csv(self.TRAIN_DATA_FILE, sep=SEPRATOR, quotechar=QUOTECHAR)
            train_df.columns = self.COLUMNS  # just for operation, names doesn't imply anything here
            train_df.head()

            # Get entity level vocab
            lines = train_df[self.ENTITY_COL].unique().tolist()
            # NUM_TAGS, tags_vocab = tf_vocab_processor(lines, ENTITY_VOCAB_FILE)
            self.NUM_TAGS, tags_vocab = naive_vocab_creater(lines, self.ENTITY_VOCAB_FILE, vocab_filter=False)
        else:
            print_info("Reusing the vocab")

    def save_preprocessed_data_info(self):
        if not os.path.exists(self.WORDS_VOCAB_FILE) and not os.path.exists(self.ENTITY_VOCAB_FILE): #TODO move this check
            # Create data level configs that is shared between model training and prediction
            info = PreprocessedDataInfo(vocab_size=self.VOCAB_SIZE,
                     num_tags=self.NUM_TAGS,
                     text_col=self.TEXT_COL,
                     entity_col=self.ENTITY_COL,
                     entity_iob_col=None,
                     train_data_file=self.TRAIN_DATA_FILE,
                     val_data_file=self.VAL_DATA_FILE,
                     test_data_file=self.TEST_DATA_FILE,
                     words_vocab_file=self.WORDS_VOCAB_FILE,
                     chars_vocab_file=self.CHARS_VOCAB_FILE,
                     entity_vocab_file=self.ENTITY_VOCAB_FILE,
                     char_2_id_map=self.char_2_id_map)

            PreprocessedDataInfo.save(info, self.OUT_DIR)
