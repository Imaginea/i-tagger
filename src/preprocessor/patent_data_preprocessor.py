import sys
sys.path.append("../")
import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.python.platform import gfile

from interfaces.preprocessor_interface import IPreprocessorInterface
from helpers.print_helper import *
from config.global_constants import *
from helpers.os_helper import check_n_makedirs


class PatentDataPreprocessor(IPreprocessorInterface):
    def __init__(self,
                 experiment_root_directory,
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
        super(PatentDataPreprocessor, self).__init__(experiment_root_directory)

        if do_run_time_config:
            self.OVER_WRITE = over_write
            self.USE_IOB = use_iob

            self.PREPROCESSED_DATA_DIR = self.EXPERIMENT_ROOT_DIR + "/" + out_dir

            self.TRAIN_CSV_PATH = train_csvs_path
            self.VAL_CSV_PATH = val_csv_path
            self.TEST_CSV_PATH = test_csv_path
            self.DB_REFERENCE_FILE = db_reference_file

            self.TEXT_COL = text_col
            self.ENTITY_COL = entity_col
            self.ENTITY_IOB_COL = entity_col + "_iob"
        else:
            self.OVER_WRITE = self.config.get_item("Options", "over_write")
            self.USE_IOB = self.config.get_item_as_boolean("Options", "use_iob_format")

            self.PREPROCESSED_DATA_DIR = self.EXPERIMENT_ROOT_DIR + "/" + self.config.get_item("OutputDirectories", "preprocessed_data_dir")

            self.TRAIN_CSV_PATH = self.config.get_item("InputDirectories", "train_csvs_path")
            self.VAL_CSV_PATH = self.config.get_item("InputDirectories", "val_csvs_path")
            self.TEST_CSV_PATH = self.config.get_item("InputDirectories", "test_csvs_path")
            self.DB_REFERENCE_FILE = self.config.get_item("InputDirectories", "db_reference_file")

            self.TEXT_COL = self.config.get_item("Schema", "text_column")
            self.ENTITY_COL = self.config.get_item("Schema", "entity_column")
            self.ENTITY_IOB_COL = self.config.get_item("Schema", "entity_column") + "_iob"

            #TODO keeping it hear as kept in other preprocessor but need to check why it is kept here
            self.EXTRA_COLS = self.config.get_item("Schema", "extra_columns")
            self.EXTRA_COLS = [str.strip(col) for col in self.EXTRA_COLS.split(",")]

        self.TRAIN_CSV_INTERMEDIATE_PATH = self.PREPROCESSED_DATA_DIR + "/train/"
        self.VAL_CSV_INTERMEDIATE_PATH = self.PREPROCESSED_DATA_DIR + "/val/"
        self.TEST_CSV_INTERMEDIATE_PATH = self.PREPROCESSED_DATA_DIR + "/test/"

        self.WORDS_VOCAB_FILE = self.PREPROCESSED_DATA_DIR + "/" + self.TEXT_COL + "_" + "vocab.tsv"
        self.CHARS_VOCAB_FILE = self.PREPROCESSED_DATA_DIR + "/" + self.TEXT_COL + "_" + "chars_vocab.tsv"
        self.ENTITY_VOCAB_FILE = self.PREPROCESSED_DATA_DIR + "/" + self.ENTITY_COL + "_vocab.tsv"

        # Make sure first two entities are text and entity followed by doc id
        # Doc Id should be taken from the extra cols
        self.COLUMNS = [self.TEXT_COL, self.ENTITY_COL]
        self.COLUMNS.extend(self.EXTRA_COLS)

        if self.USE_IOB:
            self.TRAIN_DATA_FILE = self.PREPROCESSED_DATA_DIR + "/train-doc-wise.iob"
            self.TEST_DATA_FILE = self.PREPROCESSED_DATA_DIR + "/test-doc-wise.iob"
            self.VAL_DATA_FILE = self.PREPROCESSED_DATA_DIR + "/val-doc-wise.iob"
            self.ENTITY_VOCAB_FILE = self.PREPROCESSED_DATA_DIR + "/" + self.ENTITY_IOB_COL + "_vocab.tsv"
        else:
            self.TRAIN_DATA_FILE = self.PREPROCESSED_DATA_DIR + "/train-doc-wise.io"
            self.TEST_DATA_FILE = self.PREPROCESSED_DATA_DIR + "/test-doc-wise.io"
            self.VAL_DATA_FILE = self.PREPROCESSED_DATA_DIR + "/val-doc-wise.io"
            self.ENTITY_VOCAB_FILE = self.PREPROCESSED_DATA_DIR + "/" + self.ENTITY_COL + "_vocab.tsv"

    def _create_target_directories(self):
        if os.path.exists(self.PREPROCESSED_DATA_DIR):
            if self.OVER_WRITE == "yes":
                print_info("Deletingls data folder: {}".format(self.PREPROCESSED_DATA_DIR))
                shutil.rmtree(self.PREPROCESSED_DATA_DIR)
                print_info("Recreating data folder: {}".format(self.PREPROCESSED_DATA_DIR))
                os.makedirs(self.PREPROCESSED_DATA_DIR)
                check_n_makedirs(self.TRAIN_CSV_INTERMEDIATE_PATH)
                check_n_makedirs(self.VAL_CSV_INTERMEDIATE_PATH)
                check_n_makedirs(self.TEST_CSV_INTERMEDIATE_PATH)
            else:
                print_info("Skipping preprocessing step, since the data is already available")
                return "skip"
        else:
            print_info("Creating data folder: {}".format(self.PREPROCESSED_DATA_DIR))
            os.makedirs(self.PREPROCESSED_DATA_DIR)
            check_n_makedirs(self.TRAIN_CSV_INTERMEDIATE_PATH)
            check_n_makedirs(self.VAL_CSV_INTERMEDIATE_PATH)
            check_n_makedirs(self.TEST_CSV_INTERMEDIATE_PATH)

    # ---------------------------------------------------------------------------------------
    # Following code is an updated version of above snippets to take care of consecutive same tag
    # for IOB tagging considering the actual DB records
    # ---------------------------------------------------------------------------------------

    def _match_slc(self, s, seq):
        # get list, makes zip faster
        l = s.values.tolist()
        # count how many in sequence
        k = len(seq)
        # generate numpy array of rolling values
        a = np.array(list(zip(*[l[i:] for i in range(k)])))
        # slice an array from 0 to length of a - 1 with
        # the truth values of wether all 3 in a sequence match
        p = np.arange(len(a))[(a == seq).all(1)]
        # p tracks the beginning of a match, get all subsequent
        # indices of the match as well.
        slc = np.unique(np.hstack([p + i for i in range(k)]))
        return s.iloc[slc]

    def _io_2_iob_v1(self, df, entity_col, entity_iob_col):

        tags = df[entity_col].values

        iob_tags = []

        total_len = len(tags)

        previous_tag = ""
        previous_was_zero = False
        for i in range(len(tags)):
            tag = tags[i]
            if tag == "O":
                iob_tags.append(tag)
                previous_was_zero = True
            else:
                if previous_was_zero == True:
                    iob_tags.append("B-" + tag)
                    previous_tag = tag
                    previous_was_zero = False
                else:
                    if previous_tag == tag:
                        if df["flag"].at[i] == "B":
                            iob_tags.append("B-" + tag)
                        else:
                            iob_tags.append("I-" + tag)
                        previous_tag = tag
                    else:
                        iob_tags.append("B-" + tag)
                        previous_tag = tag

        df[entity_iob_col] = iob_tags
        return df

    def _annotate_csvs(self,
                       csv_files_path,
                       db_reference_csv_file,
                       entity_col,
                       entity_iob_col,
                       out_dir):

        print_info(os.path.abspath(db_reference_csv_file))

        entity_df = pd.read_csv(db_reference_csv_file)
        # loop through the file and create a iob file with
        for csv_file in tqdm(os.listdir(csv_files_path)):
            if csv_file.endswith(".csv"):
                # print(csv_file)

                # get the current doc_id
                doc_id = csv_file.split(".")[0]

                # read the file
                file_df = pd.read_csv(os.path.join(csv_files_path, csv_file))

                # add extra column for the handling the begining of the word
                file_df = file_df.assign(flag="")

                # handle the boundary conditions
                if file_df.empty == False and len(file_df.index) > 1:
                    # get all the doc_fld_val for current document
                    doc_word_list = entity_df[entity_df.doc_id == int(doc_id)]["doc_fld_val"]

                    # convert the list of words into list of tokens. It will help in comparison
                    doc_word_tokens = list(doc_word_list.map(lambda x: x.split()))

                    # loop through the list of tokens
                    for tokens in list(doc_word_tokens):
                        # for every element get the indices in the file opened by matching the word and tokens
                        indices = self._match_slc(file_df["word"], tokens)

                        # print(indices,"\n")
                        # Now we got the indices. We need to mark the begining for every word in the file_df
                        for index_val, series_val in indices.iteritems():
                            # print(index_val,series_val,tokens[0])

                            # check if the begining of the token and start word match and set the flag ="B"
                            if tokens[0] == series_val:
                                file_df.at[index_val, 'flag'] = "B"

                    # call the IOB function to use the flag and mark the ambiguous cases
                    file_df = self._io_2_iob_v1(file_df, entity_col, entity_iob_col)

                    # incase you want to drop the flag column
                    # file_df = file_df.drop('flag', axis=1)
                    file_df.to_csv(os.path.join(out_dir, csv_file), index=False)

    def _csv_to_conll_format_v1(self,
                                csv_files_path,
                                outfilename,
                                text_col,
                                entity_col,
                                entity_iob_col,
                                use_iob,
                                unknown_token):
        '''

        :param csv_files_path:
        :param text_col:
        :param entity_col:
        :param outfilename:
        :param unknown_token:
        :return:
        '''

        num_records = 0

        if use_iob:
            self.COLUMNS.append(self.ENTITY_IOB_COL)

        with gfile.Open(outfilename, 'wb') as file:
            for csv_file in tqdm(os.listdir(csv_files_path)):
                csv_file = os.path.join(csv_files_path, csv_file)
                df = pd.read_csv(csv_file).fillna(unknown_token)


                # if it is very first record. Take the column names as header for the csv files
                if num_records == 0:
                    file.write("{}\n".format(" ".join(self.COLUMNS)))

                try:
                    values = df[self.COLUMNS].values  # TODO 1

                    # ['word', 'entity_name', 'doc_id', 'x_cord', 'y_cord', 'pg_number', %IOB_COL]
                    tot_len = len(values)
                    if tot_len > 10:
                        num_records += 1
                        #TODO beautify this
                        #Note: Make sure to have the text col and entity as first two cols

                        #take the first row of value and set it as start
                        start_str = ["<START>", "O"]
                        for s_k in(values[0][2:]):
                            start_str.append(str(s_k))

                        file.write("{}\n".format(" ".join(start_str)))

                        for i in range(tot_len):
                            file_str = []
                            for f_k in (values[i]):
                                file_str.append(str(f_k))
                            file.write("{}\n".format(" ".join(file_str)))  # TODO 1

                        # take the last row of value and set it as end
                        end_str = ["<END>", "O"]
                        for e_k in (values[i][2:]):
                            end_str.append(str(e_k))

                        file.write("{}\n".format(" ".join(end_str)))
                        file.write("\n")
                except Exception as e:
                    print("Error procesing : {} in {}".format(e, csv_file))

        print("Total of {} records got written!".format(num_records))

        # Return column details to assist vocab creation
        return self.COLUMNS

    def _prepare_data(self):
        print_info("Preparing train data...")

        self._annotate_csvs(csv_files_path=self.TRAIN_CSV_PATH,
                      db_reference_csv_file=self.DB_REFERENCE_FILE,
                      entity_col=self.ENTITY_COL,
                      entity_iob_col=self.ENTITY_IOB_COL,
                      out_dir=self.TRAIN_CSV_INTERMEDIATE_PATH)

        self._csv_to_conll_format_v1(csv_files_path=self.TRAIN_CSV_INTERMEDIATE_PATH,
                                     outfilename=self.TRAIN_DATA_FILE,
                                     text_col=self.TEXT_COL,
                                     entity_col=self.ENTITY_COL,
                                     entity_iob_col=self.ENTITY_IOB_COL,
                                     use_iob=self.USE_IOB,
                                     unknown_token=UNKNOWN_WORD)

        print_info("Preparing test data...")

        self._annotate_csvs(csv_files_path=self.TEST_CSV_PATH,
                      db_reference_csv_file=self.DB_REFERENCE_FILE,
                      entity_col=self.ENTITY_COL,
                      entity_iob_col=self.ENTITY_IOB_COL,
                      out_dir=self.TEST_CSV_INTERMEDIATE_PATH)

        self.COLUMNS = self._csv_to_conll_format_v1(csv_files_path=self.TEST_CSV_INTERMEDIATE_PATH,
                                               outfilename=self.TEST_DATA_FILE,
                                               text_col=self.TEXT_COL,
                                               entity_col=self.ENTITY_COL,
                                               entity_iob_col=self.ENTITY_IOB_COL,
                                               use_iob=self.USE_IOB,
                                               unknown_token=UNKNOWN_WORD)

        print_info("Preparing validation data...")

        self._annotate_csvs(csv_files_path=self.VAL_CSV_PATH,
                      db_reference_csv_file=self.DB_REFERENCE_FILE,
                      entity_col=self.ENTITY_COL,
                      entity_iob_col=self.ENTITY_IOB_COL,
                      out_dir=self.VAL_CSV_INTERMEDIATE_PATH)

        self.COLUMNS = self._csv_to_conll_format_v1(csv_files_path=self.VAL_CSV_INTERMEDIATE_PATH,
                                               outfilename=self.VAL_DATA_FILE,
                                               text_col=self.TEXT_COL,
                                               entity_col=self.ENTITY_COL,
                                               entity_iob_col=self.ENTITY_IOB_COL,
                                               use_iob=self.USE_IOB,
                                               unknown_token=UNKNOWN_WORD)

