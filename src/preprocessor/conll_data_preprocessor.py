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

            self.TRAIN_DF_PATH = self.config.get_item("InputDirectories", "train_txt_path")
            self.VAL_DF_PATH = self.config.get_item("InputDirectories", "val_txt_path")
            self.TEST_DF_PATH = self.config.get_item("InputDirectories", "test_txt_path")

    def _create_target_directories(self):
        if os.path.exists(self.DATA_OUT_DIR):
            if self.OVER_WRITE == "yes":
                print_info("Deletingls data folder: {}".format(self.DATA_OUT_DIR))
                shutil.rmtree(self.DATA_OUT_DIR)
                print_info("Recreating data folder: {}".format(self.DATA_OUT_DIR))
                os.makedirs(self.DATA_OUT_DIR)
            else:
                print_info("Skipping preprocessing step, since the data might already be available")
                exit(0)
        else:
            print_info("Creating data folder: {}".format(self.DATA_OUT_DIR))
            os.makedirs(self.DATA_OUT_DIR)

    def _conll_to_csv(self, txt_file_path, out_dir):
        '''
        Function to convert CoNLL 2003 data set text files into CSV file for each 
        example/statement.
        :param txt_file_path: Input text file path
        :param out_dir: Output directory to store CSV files
        :return: 
        '''
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Read the text file
        df = pd.read_csv(txt_file_path,
                         sep=" ",
                         skip_blank_lines=False,
                         header=None).fillna(UNKNOWN_WORD)

        # Filter out the DOCSTART lines
        df = df[~df[0].str.contains("DOCSTART")]

        current_file = []

        for i in tqdm(range(len(df))):
            row = df.values[i]
            if row[0] != UNKNOWN_WORD:
                current_file.append(row)
            else:
                # Consider dumping files with size 2
                if len(current_file) > 2:
                    current_file = pd.DataFrame(current_file)
                    current_file.to_csv(out_dir + "/{}.csv".format(i), index=False)
                    current_file = []

    def _prepare_data(self):
        print_info("Preprocessing the train data...")
        self._conll_to_csv(self.TRAIN_DF_PATH, self.TRAIN_OUT_PATH)

        print_info("Preprocessing the test data...")
        self._conll_to_csv(self.TEST_DF_PATH, self.TEST_OUT_PATH)

        print_info("Preprocessing the validation data...")
        self._conll_to_csv(self.VAL_DF_PATH, self.VAL_OUT_PATH)
