import pdb
import sys

from helpers.os_helper import check_n_makedirs

sys.path.append("../")
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import pickle
import ntpath
import traceback
import tensorflow as tf
from interfaces.data_iterator import IDataIterator
from helpers.tf_hooks.data_initializers import DataIteratorInitializerHook
from overrides import overrides
from helpers.print_helper import *
from config.global_constants import *
from tensorflow.python.platform import gfile
from interfaces.two_features_interface import ITextFeature

class CsvDataIterator(IDataIterator, ITextFeature):
    def __init__(self, experiment_dir, batch_size):
        IDataIterator.__init__(self, experiment_dir, batch_size)
        ITextFeature.__init__(self)

    def __pad_sequences(self, sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with

        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    def _pad_sequences(self, sequences, pad_tok, nlevels, MAX_WORD_LENGTH=MAX_WORD_LENGTH):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids

        Returns:
            a list of list where each sublist has same length

        """
        if nlevels == 1:
            sequence_padded = []
            sequence_length = []
            max_length = max(map(lambda x: len(x.split(SEPERATOR)), sequences))
            # sequence_padded, sequence_length = _pad_sequences(sequences,
            #                                                   pad_tok, max_length)
            # breaking the code to pad the string instead on its ids
            for seq in sequences:
                current_length = len(seq.split(SEPERATOR))
                diff = max_length - current_length
                pad_data = pad_tok * diff
                sequence_padded.append(seq + pad_data)
                sequence_length.append(max_length)  # assumed

                # print_info(sequence_length)
                #TODO Hey mages can you elaborate the use of this function ?
        elif nlevels == 2:
            # max_length_word = max([max(map(lambda x: len(x), seq))
            #                        for seq in sequences])
            sequence_padded, sequence_length = [], []
            for seq in tqdm(sequences):
                # all words are same length now
                sp, sl = self.__pad_sequences(seq, pad_tok, MAX_WORD_LENGTH)
                sequence_padded += [sp]
                sequence_length += [sl]

            max_length_sentence = max(map(lambda x: len(x), sequences))
            sequence_padded, _ = self.__pad_sequences(sequence_padded,
                                                     [pad_tok] * MAX_WORD_LENGTH,
                                                     max_length_sentence)
            sequence_length, _ = self.__pad_sequences(sequence_length, 0,
                                                     max_length_sentence)

        return np.array(sequence_padded), sequence_length

    def _make_seq_pair(self, df_files_path, char_2_id_map, use_char_embd):
        '''
        Reads the CoNLL text file and makes Sentence-Tags pair for NN model
        :param df_files_path:
        :param word_col:
        :param tag_col:
        :param empty_line_filler:
        :return:
        '''

        self.TEXT_COL = self.preprocessed_data_info.TEXT_COL
        self.ENTITY_IOB_COL = self.preprocessed_data_info.ENTITY_IOB_COL

        list_text = []
        list_char_ids = []
        list_tag = []

        # [feature1 ,feature2, label]
        sentence_feature1 = []
        char_ids_feature2 = []
        tag_label = []

        for df_file in tqdm(os.listdir(df_files_path)):

            # Make the container empty
            list_text = []
            list_char_ids = []
            list_tag = []

            df_file = os.path.join(df_files_path, df_file)

            if df_file.endswith(".csv"): #TODO start and stop tags
                df = pd.read_csv(df_file).fillna(UNKNOWN_WORD)
            elif df_file.endswith(".json"):
                df = pd.read_json(df_file).filla(UNKNOWN_WORD)

            list_text = df[self.TEXT_COL].astype(str).values.tolist()
            list_char_ids = [[char_2_id_map.get(c, 0) for c in str(word)] for word in list_text]
            list_tag = df[self.ENTITY_IOB_COL].astype(str).values.tolist()

            sentence_feature1.append("{}".format(SEPERATOR).join(list_text))
            char_ids_feature2.append(list_char_ids)
            tag_label.append("{}".format(SEPERATOR).join(list_tag))


        if use_char_embd:
            sentence_feature1, seq_length = self._pad_sequences(sentence_feature1,
                                                                nlevels=1,
                                                                pad_tok="{}{}".format(SEPERATOR, PAD_WORD))  # space is used so that it can append to the string sequence
            sentence_feature1 = np.array(sentence_feature1)

            char_ids_feature2, seq_length = self._pad_sequences(char_ids_feature2, nlevels=2, pad_tok=int(PAD_CHAR_ID))
            char_ids_feature2 = np.array(char_ids_feature2)
            seq_length = np.array(seq_length)
            # print_warn(seq_length.shape)
            # exit()
            tag_label, seq_length = self._pad_sequences(tag_label,
                                                        nlevels=1,
                                                        pad_tok="{}{}".format(SEPERATOR, PAD_WORD))
            tag_label = np.array(tag_label)

            return sentence_feature1, char_ids_feature2, tag_label

        else:
            sentence_feature1 = np.array(sentence_feature1)
            tag_label = np.array(tag_label)
            return sentence_feature1, None, tag_label

    #######################################################################################
    #               TF Data Graph Operations
    #######################################################################################

    def _setup_input_graph2(self, text_features, char_ids, labels, batch_size,
                            # num_epocs,
                            use_char_embd=False,
                            is_eval=False,
                            shuffle=True,
                            scope='train-data'):

        iterator_initializer_hook = DataIteratorInitializerHook()

        char_ids = np.array(char_ids)

        tf.logging.info("text_features.shape: =====> {}".format(text_features.shape))
        tf.logging.info("numeric_features.shape: =====> {}".format(char_ids.shape))
        tf.logging.info("labels.shape: =====> {}".format(labels.shape))

        # pdb.set_trace()

        def inputs():
            """Returns training set as Operations.

            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            with tf.name_scope(scope):

                # Define placeholders
                text_features_placeholder = tf.placeholder(tf.string, text_features.shape, name="sentence")
                if use_char_embd:
                    char_ids_placeholder = tf.placeholder(tf.int32, [None, None, MAX_WORD_LENGTH], name="char_ids")
                labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name="label")

                # Build dataset iterator
                if use_char_embd:
                    dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: text_features_placeholder,
                                                                      self.FEATURE_2_NAME: char_ids_placeholder},
                                                                      labels_placeholder))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: text_features_placeholder},
                                                                  labels_placeholder))
                if is_eval:
                    dataset = dataset.repeat(1)
                else:
                    dataset = dataset.repeat(None)  # Infinite iterations

                if shuffle:
                    dataset = dataset.shuffle(buffer_size=10000)
                dataset = dataset.batch(batch_size)
                iterator = dataset.make_initializable_iterator()

                # Set runhook to initialize iterator
                if use_char_embd:
                    iterator_initializer_hook.iterator_initializer_func = \
                        lambda sess: sess.run(
                            iterator.initializer,
                            feed_dict={text_features_placeholder: text_features,
                                       char_ids_placeholder: char_ids,
                                       labels_placeholder: labels})
                else:
                    iterator_initializer_hook.iterator_initializer_func = \
                        lambda sess: sess.run(
                            iterator.initializer,
                            feed_dict={text_features_placeholder: text_features,
                                       labels_placeholder: labels})

                next_features, next_label = iterator.get_next()

                # Return batched (features, labels)
                return next_features, next_label

        # Return function and hook
        return inputs, iterator_initializer_hook

    def predict_inputs(self, features, char_ids, batch_size=1, scope='test-data'):
        """Returns test set as Operations.
        Returns:
            (features, ) Operations that iterate over the test set.
        """
        # Convert raw sentence into a lisr, since TF works on only list/matrix
        if not isinstance(features, list):
            features = [features]

        def inputs():
            with tf.name_scope(scope):
                docs = tf.constant(features, dtype=tf.string)
                dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: docs,
                                                               self.FEATURE_2_NAME: char_ids},))
                dataset.repeat(1)
                # Return as iteration in batches of 1
                return dataset.batch(batch_size).make_one_shot_iterator().get_next()

        return inputs


    @overrides
    def setup_train_input_graph(self):
        train_sentences, train_char_ids, train_ner_tags = \
            self._make_seq_pair(df_files_path=self.preprocessed_data_info.TRAIN_FILES_PATH,
                                char_2_id_map=self.preprocessed_data_info.char_2_id_map,
                                use_char_embd=True) #TODO

        self.NUM_TRAINING_SAMPLES = train_sentences.shape[0] #TODO

        self._train_data_input_fn, self._train_data_init_hook = self._setup_input_graph2(text_features=train_sentences,
                                                                                     char_ids=train_char_ids,
                                                                                     labels=train_ner_tags,
                                                                                     batch_size=self.BATCH_SIZE,
                                                                                     use_char_embd=True) #TODO
    @overrides
    def setup_val_input_graph(self):
        val_sentences, val_char_ids, val_ner_tags = \
            self._make_seq_pair(df_files_path=self.preprocessed_data_info.VAL_FILES_PATH,
                                char_2_id_map=self.preprocessed_data_info.char_2_id_map,
                                use_char_embd=True) #TODO
        self._val_data_input_fn, self._val_data_init_hook = self._setup_input_graph2(text_features=val_sentences,
                                                                                     char_ids=val_char_ids,
                                                                                     labels=val_ner_tags,
                                                                                     batch_size=self.BATCH_SIZE,
                                                                                     use_char_embd=True,
                                                                                     is_eval=True) #TODO

    def get_tags(self, estimator, sentence, char_ids, tag_vocab_tsv):

        with gfile.Open(tag_vocab_tsv, 'r') as file:
            ner_vocab = list(map(lambda x: x.strip(), file.readlines()))
            tags_vocab = {id_num: tag for id_num, tag in enumerate(ner_vocab)}

        predictions = []
        test_input_fn = self.predict_inputs(sentence, char_ids)
        predict_fn = estimator.predict(input_fn=test_input_fn)


        for predict in predict_fn:
            predictions.append(predict)

        predicted_id = []
        confidence = []

        for each_prediction in predictions:
            for tag_score in each_prediction["confidence"]:
                confidence.append(tag_score)
            for tag_id in each_prediction["viterbi_seq"]:
                predicted_id.append(tags_vocab[tag_id])
            top_3_predicted_indices = each_prediction["top_3_indices"]
            top_3_predicted_confidence = each_prediction["top_3_confidence"]

            # print(top_3_predicted_indices)

            pred_1 = top_3_predicted_indices[:, 0:1].flatten()
            pred_1 = list(map(lambda x: tags_vocab[x], pred_1))
            # print(pred_1)
            # print(predicted_id)

            pred_2 = top_3_predicted_indices[:, 1:2].flatten()
            pred_2 = list(map(lambda x: tags_vocab[x], pred_2))
            pred_3 = top_3_predicted_indices[:, 2:].flatten()
            pred_3 = list(map(lambda x: tags_vocab[x], pred_3))

            pred_1_confidence = top_3_predicted_confidence[:, 0:1]
            pred_2_confidence = top_3_predicted_confidence[:, 1:2]
            pred_3_confidence = top_3_predicted_confidence[:, 2:]

        return predicted_id, confidence, pred_1, pred_1_confidence, pred_2, pred_2_confidence, \
               pred_3, pred_3_confidence

    def predict_on_test_file(self, estimator, df):
        sentence = ("{}".format(SEPERATOR).
                    join(df[self.preprocessed_data_info.TEXT_COL].astype(str).values))

        char_ids = [[self.preprocessed_data_info.char_2_id_map.get(c, 0)
                     for c in word] for word in sentence.split(SEPERATOR)]

        char_ids, char_ids_length = self._pad_sequences([char_ids], pad_tok=int(PAD_CHAR_ID), nlevels=2)
        # TODO add batch support
        predicted_tags, confidence, \
        pred_1, pred_1_confidence, \
        pred_2, pred_2_confidence, \
        pred_3, pred_3_confidence = self.get_tags(estimator,
                                                  sentence,
                                                  char_ids,
                                                  self.preprocessed_data_info.ENTITY_VOCAB_FILE)

        df["predictions"] = predicted_tags
        df["confidence"] = confidence
        df["pred_1"] = pred_1
        df["pred_1_confidence"] = pred_1_confidence
        df["pred_2"] = pred_2
        df["pred_2_confidence"] = pred_2_confidence
        df["pred_3"] = pred_3
        df["pred_3_confidence"] = pred_3_confidence

        return df

    def predict_on_test_files(self, estimator, csv_files_path):

        failed_csvs = []

        for csv_file in tqdm(os.listdir(csv_files_path)):
            csv_file = os.path.join(csv_files_path, csv_file)
            if csv_file.endswith(".csv"):
                sentence = ""
                try:
                    print_info("processing ====> {}".format(csv_file))
                    df = pd.read_csv(csv_file).fillna(UNKNOWN_WORD)

                    df = self.predict_on_test_file(df)

                    out_dir = estimator.model_dir +"/predictions/"
                    check_n_makedirs(out_dir)
                    df.to_csv(out_dir + ntpath.basename(csv_file), index=False)
                except Exception as e:
                    print_error(traceback.print_exc())
                    failed_csvs.append(csv_file)
                    print_warn("Failed processing ====> {}".format(csv_file))
                    pdb.set_trace()

        print_error(failed_csvs)
        return out_dir


    def predict_on_text(self, estimator, sentence):

        # Convert space delimited text to a sentence delimited  by `SEPERATOR`
        sentence = sentence.split()
        sentence = "{}".format(SEPERATOR).join(sentence)

        # Trailing by 213, Somerset got a solid start to their second innings before Simmons stepped in to bundle them out for 174.
        char_ids = [[self.preprocessed_data_info.char_2_id_map.get(c, 0) for c in word] for word in sentence.split(SEPERATOR)]
        char_ids, char_ids_length = self._pad_sequences([char_ids], pad_tok=0, nlevels=2)

        predicted_tags, confidence, pred_1, pred_1_confidence, pred_2, pred_2_confidence, \
        pred_3, pred_3_confidence = self.get_tags(estimator, sentence, char_ids, self.preprocessed_data_info.ENTITY_VOCAB_FILE)

        print_info(predicted_tags)
        return predicted_tags
