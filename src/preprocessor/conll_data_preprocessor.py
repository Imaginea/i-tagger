import sys
sys.path.append("../")
import os
import shutil
import pandas as pd
from tqdm import tqdm

from interfaces.preprocessor_interface import IPreprocessorInterface
from helpers.print_helper import *
from config.global_constants import *
from helpers.os_helper import copytree

from nlp.spacy_helper import naive_vocab_creater, get_char_vocab, vocab_to_tsv
from config.preprocessed_data_info import PreprocessedDataInfo

class CoNLLDataPreprocessor(IPreprocessorInterface):
    def __init__(self,
                 experiment_root_directory,
                 over_write=None,
                 train_df_path=None,
                 val_df_path=None,
                 test_df_path=None,
                 text_col=None,
                 entity_col=None,
                 do_run_time_config=False):
        '''
        
        :param over_write: 
        :param use_iob: 
        :param out_dir: 
        :param train_df_path: 
        :param val_df_path: 
        :param test_df_path: 
        :param text_col: 
        :param entity_col: 
        :param do_run_time_config: Enable this to use constructor params or 
        by default it uses config/patent_data_preprocessor.ini config
        '''
        super(CoNLLDataPreprocessor, self).__init__(experiment_root_directory)

        if do_run_time_config:
            self.OVER_WRITE = over_write

            self.TRAIN_DF_PATH = train_df_path
            self.VAL_DF_PATH = val_df_path
            self.TEST_DF_PATH = test_df_path

            self.TEXT_COL = text_col
            self.ENTITY_COL = entity_col
        else:
            self.OVER_WRITE = self.config.get_item("Options", "over_write")

            self.TRAIN_DF_PATH = self.config.get_item("InputDirectories", "train_csvs_path")
            self.VAL_DF_PATH = self.config.get_item("InputDirectories", "val_csvs_path")
            self.TEST_DF_PATH = self.config.get_item("InputDirectories", "test_csvs_path")

            self.TEXT_COL = self.config.get_item("Schema", "text_column")
            self.ENTITY_COL = self.config.get_item("Schema", "entity_column")

        self.TRAIN_OUT_PATH = self.OUT_DIR + "/train/"
        self.VAL_OUT_PATH = self.OUT_DIR + "/val/"
        self.TEST_OUT_PATH = self.OUT_DIR + "/test/"

        self.WORDS_VOCAB_FILE = self.OUT_DIR + "/" + self.TEXT_COL + "_" + "vocab.tsv"
        self.CHARS_VOCAB_FILE = self.OUT_DIR + "/" + self.TEXT_COL + "_" + "chars_vocab.tsv"
        self.ENTITY_VOCAB_FILE = self.OUT_DIR + "/" + self.ENTITY_COL + "_vocab.tsv"

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
        print_info("No preprocessing, just copying the train data...")
        copytree(src=self.TRAIN_DF_PATH, dst=self.TRAIN_OUT_PATH)

        print_info("No preprocessing, just copying the test data...")
        copytree(src=self.TEST_DF_PATH, dst=self.TEST_OUT_PATH)

        print_info("No preprocessing, just copying the validation data...")
        copytree(src=self.VAL_DF_PATH, dst=self.VAL_OUT_PATH)


    def extract_vocab(self):
        if not os.path.exists(self.WORDS_VOCAB_FILE) or not os.path.exists(self.ENTITY_VOCAB_FILE):
            print_info("Preparing the vocab for the text col: {}".format(self.TEXT_COL))

            lines = set()
            entities = set()

            for df_file in tqdm(os.listdir(self.TRAIN_DF_PATH)):
                df_file = os.path.join(self.TRAIN_DF_PATH, df_file)
                if df_file.endswith(".csv"):
                    df = pd.read_csv(df_file).fillna(UNKNOWN_WORD)
                elif df_file.endswith(".json"):
                    df = pd.read_json(df_file).filla(UNKNOWN_WORD)
                lines.update(set(df[self.TEXT_COL].values.tolist()))
                entities.update(set(df[self.ENTITY_COL].values.tolist()))

            # VOCAB_SIZE, words_vocab = tf_vocab_processor(lines, WORDS_VOCAB_FILE)
            self.VOCAB_SIZE, words_vocab = naive_vocab_creater(lines, self.WORDS_VOCAB_FILE, use_nlp=True)

            # Get char level vocab
            words_chars_vocab = [PAD_CHAR, UNKNOWN_CHAR]
            _vocab = get_char_vocab(words_vocab)
            words_chars_vocab.extend(_vocab)

            # Create char2id map
            vocab_to_tsv(words_chars_vocab, self.CHARS_VOCAB_FILE)
            self.char_2_id_map = {c: i for i, c in enumerate(words_chars_vocab)}

            print_info("Preparing the vocab for the entity col: {}".format(self.ENTITY_COL))

            # NUM_TAGS, tags_vocab = tf_vocab_processor(lines, ENTITY_VOCAB_FILE)
            self.NUM_TAGS, tags_vocab = naive_vocab_creater(entities, self.ENTITY_VOCAB_FILE, use_nlp=False)
        else:
            print_info("Reusing the vocab")


    def save_preprocessed_data_info(self):
        if not PreprocessedDataInfo.is_file_exists(self.OUT_DIR):
            # Create data level configs that is shared between model training and prediction
            info = PreprocessedDataInfo(vocab_size=self.VOCAB_SIZE,
                                        num_tags=self.NUM_TAGS,
                                        text_col=self.TEXT_COL,
                                        entity_col=self.ENTITY_COL,
                                        entity_iob_col=self.ENTITY_COL, #same in this case
                                        train_files_path=self.TRAIN_OUT_PATH,
                                        val_files_path=self.VAL_OUT_PATH,
                                        test_path_files=self.TEST_OUT_PATH,
                                        words_vocab_file=self.WORDS_VOCAB_FILE,
                                        chars_vocab_file=self.CHARS_VOCAB_FILE,
                                        entity_vocab_file=self.ENTITY_VOCAB_FILE,
                                        char_2_id_map=self.char_2_id_map)

            PreprocessedDataInfo.save(info, self.EXPERIMENT_ROOT_DIR)
