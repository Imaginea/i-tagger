import sys

from helpers.os_helper import check_n_makedirs

sys.path.append("../")
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import ntpath

import tensorflow as tf
from interfaces.data_iterator import IDataIterator
from helpers.tf_hooks.data_initializers import DataIteratorInitializerHook
from overrides import overrides
from helpers.print_helper import *
from config.global_constants import *
from interfaces.two_features_interface import IPostionalFeature
from nlp.spacy_helper import naive_vocab_creater, get_char_vocab, vocab_to_tsv


class PositionalPatentDataIterator(IDataIterator, IPostionalFeature):
    def __init__(self, experiment_dir, batch_size):
        IDataIterator.__init__(self, "positional_patent_data_iterator", experiment_dir, batch_size)
        IPostionalFeature.__init__(self)

        self.POSITIONAL_COL = self.config.get_item("Schema", "positional_column")
        self.extract_vocab()

    def extract_vocab(self):
        if not os.path.exists(self.WORDS_VOCAB_FILE) \
                or not os.path.exists(self.ENTITY_VOCAB_FILE) \
                or not os.path.exists(self.CHARS_VOCAB_FILE):
            print_info("Preparing the vocab for the text col: {}".format(self.TEXT_COL))

            lines = set()
            entities = set()

            for df_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH), desc="merging lines"):
                df_file = os.path.join(self.TRAIN_FILES_IN_PATH, df_file)
                if df_file.endswith(".csv"):
                    df = pd.read_csv(df_file).fillna(UNKNOWN_WORD)
                elif df_file.endswith(".json"):
                    df = pd.read_json(df_file).filla(UNKNOWN_WORD)
                lines.update(set(df[self.TEXT_COL].values.tolist()))
                entities.update(set(df[self.ENTITY_COL].values.tolist()))

            self.VOCAB_SIZE, words_vocab = naive_vocab_creater(lines=lines, out_file_name=self.WORDS_VOCAB_FILE,
                                                               use_nlp=True)

            print_info("Preparing the character vocab for the text col: {}".format(self.TEXT_COL))

            # Get char level vocab
            char_vocab = [PAD_CHAR, UNKNOWN_CHAR]
            _vocab = get_char_vocab(words_vocab)
            char_vocab.extend(_vocab)

            # Create char2id map
            self.char_2_id_map = vocab_to_tsv(vocab_list=char_vocab, out_file_name=self.CHARS_VOCAB_FILE)
            self.CHAR_VOCAB_SIZE = len(self.char_2_id_map)

            print_info("Preparing the vocab for the entity col: {}".format(self.ENTITY_COL))

            # NUM_TAGS, tags_vocab = tf_vocab_processor(lines, ENTITY_VOCAB_FILE)
            self.NUM_TAGS, tags_vocab = naive_vocab_creater(lines=entities, out_file_name=self.ENTITY_VOCAB_FILE,
                                                            use_nlp=False)
        else:
            print_info("Reusing the vocab")
            self.VOCAB_SIZE, words_vocab = naive_vocab_creater(lines=None, out_file_name=self.WORDS_VOCAB_FILE,
                                                               use_nlp=None)
            self.char_2_id_map = vocab_to_tsv(out_file_name=self.CHARS_VOCAB_FILE, vocab_list=None)
            self.CHAR_VOCAB_SIZE = len(self.char_2_id_map)

            self.NUM_TAGS, tags_vocab = naive_vocab_creater(lines=None, out_file_name=self.ENTITY_VOCAB_FILE,
                                                            use_nlp=False)
            self.TAGS_2_ID = {id_num: tag for id_num, tag in enumerate(tags_vocab)}

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
                # TODO Hey mages can you elaborate the use of this function ?
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

    def pad_position(self, sequences, pad_tok=[0, 0, 0]):
        """
        Args:
            sequences: a list of [x,y,z] positions eg:  [[], [(1,2,3),(4,5,6)],[(7,8,9)], []]
            pad_tok: default x,y,z postion
        Returns:
            a list of list where each sublist has same length

        """
        sequence_padded = []
        max_length = max(map(lambda x: len(x), sequences))
        for seq in sequences:
            copy_seq = seq
            current_length = len(seq)
            diff = max_length - current_length
            pad_data = [pad_tok for _ in range(diff)]
            copy_seq.extend(pad_data)
            sequence_padded.append(copy_seq)
        return np.array(sequence_padded), max_length

    def _make_seq_pair(self, df_files_path, char_2_id_map, use_char_embd):
        '''
        Reads the CoNLL text file and makes Sentence-Tags pair for NN model
        :param df_files_path:
        :param word_col:
        :param tag_col:
        :param empty_line_filler:
        :return:
        '''

        list_text = []
        list_positions = []
        list_char_ids = []
        list_tag = []

        # [feature1 ,feature2, label]
        sentence_feature1 = []
        char_ids_feature2 = []
        positional_feature3 = []
        tag_label = []

        for df_file in tqdm(os.listdir(df_files_path)):
            # Make the container empty
            list_text = []
            list_char_ids = []
            list_positions = []
            list_tag = []

            df_file = os.path.join(df_files_path, df_file)

            if df_file.endswith(".csv"):  # TODO start and stop tags
                df = pd.read_csv(df_file).fillna(UNKNOWN_WORD)
            elif df_file.endswith(".json"):
                df = pd.read_json(df_file).filla(UNKNOWN_WORD)

            list_text = df[self.TEXT_COL].astype(str).values.tolist()
            list_char_ids = [[char_2_id_map.get(c, 0) for c in str(word)] for word in list_text]
            list_positions = df[self.POSITIONAL_COL.split(",")].values.tolist()
            list_tag = df[self.ENTITY_COL].astype(str).values.tolist()

            sentence_feature1.append("{}".format(SEPERATOR).join(list_text))
            char_ids_feature2.append(list_char_ids)
            positional_feature3.append(list_positions)
            tag_label.append("{}".format(SEPERATOR).join(list_tag))

        positional_feature3, seq_length = self.pad_position(positional_feature3)

        if use_char_embd:
            sentence_feature1, seq_length = self._pad_sequences(sentence_feature1,
                                                                nlevels=1,
                                                                pad_tok="{}{}".format(SEPERATOR,
                                                                                      PAD_WORD))  # space is used so that it can append to the string sequence
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

            return sentence_feature1, char_ids_feature2, positional_feature3, tag_label

        else:
            sentence_feature1 = np.array(sentence_feature1)
            tag_label = np.array(tag_label)
            return sentence_feature1, None, positional_feature3, tag_label

    #######################################################################################
    #               TF Data Graph Operations
    #######################################################################################

    def _setup_input_graph2(self, text_features, positional_features, char_ids, labels, batch_size,
                            # num_epocs,
                            use_char_embd=False,
                            is_eval=False,
                            shuffle=True,
                            scope='train-data'):
        """Return the input function to get the training data.

        Args:
            batch_size (int): Batch size of training iterator that is returned
                              by the input function.
            mnist_data (Object): Object holding the loaded mnist data.

        Returns:
            (Input function, IteratorInitializerHook):
                - Function that returns (features, labels) when called.
                - Hook to initialise input iterator.
        """
        iterator_initializer_hook = DataIteratorInitializerHook()

        tf.logging.info("text_features.shape: =====> {}".format(text_features.shape))
        tf.logging.info("positional_features.shape: =====> {}".format(positional_features.shape))
        # tf.logging.info("numeric_features.shape: =====> {}".format(char_ids.shape))
        tf.logging.info("labels.shape: =====> {}".format(labels.shape))

        tf.logging.info("text_features.shape: =====> {}".format(type(text_features)))
        tf.logging.info("positional_features.type: =====> {}".format(type(positional_features)))

        tf.logging.info("numeric_features.type: =====> {}".format(type(char_ids)))
        char_ids = np.array(char_ids)
        tf.logging.info("labels.type: =====> {}".format(type(labels)))

        # tf.logging.info("numeric_features.shape: =====> {}".format((char_ids)))

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
                positional_features_placeholder = tf.placeholder(tf.float32, positional_features.shape, name="position")

                if use_char_embd:
                    char_ids_placeholder = tf.placeholder(tf.int32, [None, None, MAX_WORD_LENGTH], name="char_ids")
                labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name="label")

                # Build dataset iterator
                if use_char_embd:
                    dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: text_features_placeholder,
                                                                   self.FEATURE_2_NAME: char_ids_placeholder,
                                                                   self.FEATURE_3_NAME: positional_features_placeholder},
                                                                  labels_placeholder))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: text_features_placeholder,
                                                                   self.FEATURE_3_NAME: positional_features_placeholder},
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
                                       positional_features_placeholder: positional_features,
                                       labels_placeholder: labels})
                else:
                    iterator_initializer_hook.iterator_initializer_func = \
                        lambda sess: sess.run(
                            iterator.initializer,
                            feed_dict={text_features_placeholder: text_features,
                                       positional_features_placeholder: positional_features,
                                       labels_placeholder: labels})

                next_features, next_label = iterator.get_next()

                # Return batched (features, labels)
                return next_features, next_label

        # Return function and hook
        return inputs, iterator_initializer_hook

    @overrides
    def setup_train_input_graph(self):
        train_sentences, train_char_ids, train_positions, train_ner_tags = \
            self._make_seq_pair(df_files_path=self.TRAIN_FILES_IN_PATH,
                                char_2_id_map=self.char_2_id_map,
                                use_char_embd=True)  # TODO
        self.NUM_TRAINING_SAMPLES = train_sentences.shape[0]  # TODO

        self._train_data_input_fn, self._train_data_init_hook = self._setup_input_graph2(text_features=train_sentences,
                                                                                         positional_features=train_positions,
                                                                                         char_ids=train_char_ids,
                                                                                         labels=train_ner_tags,
                                                                                         batch_size=self.BATCH_SIZE,
                                                                                         use_char_embd=True)  # TODO

    @overrides
    def setup_val_input_graph(self):
        val_sentences, val_char_ids, eval_positions, val_ner_tags = \
            self._make_seq_pair(df_files_path=self.VAL_FILES_IN_PATH,
                                char_2_id_map=self.char_2_id_map,
                                use_char_embd=True)  # TODO
        self._val_data_input_fn, self._val_data_init_hook = self._setup_input_graph2(text_features=val_sentences,
                                                                                     positional_features=eval_positions,
                                                                                     char_ids=val_char_ids,
                                                                                     labels=val_ner_tags,
                                                                                     batch_size=self.BATCH_SIZE,
                                                                                     use_char_embd=True,
                                                                                     is_eval=True)  # TODO

    def setup_predict_graph(self, features, positional_features, char_ids, batch_size=12, scope='test-data'):
        """Returns test set as Operations.
        Returns:
            (features, ) Operations that iterate over the test set.
        """
        # Convert raw sentence into a lisr, since TF works on only list/matrix
        if not isinstance(features, list):
            features = [features]

        # TODO mages why this is not below doc in scope below
        positional_features = tf.constant(positional_features, dtype=tf.float32)

        def inputs():
            with tf.name_scope(scope):
                docs = tf.constant(features, dtype=tf.string)

                dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: docs,
                                                               self.FEATURE_2_NAME: char_ids,
                                                               self.FEATURE_3_NAME: positional_features},))
                dataset.repeat(1)
                # Return as iteration in batches of 1
                return dataset.batch(batch_size).make_one_shot_iterator().get_next()

        return inputs

    def get_tags(self, estimator, sentence, positions, char_ids):

        predictions = []
        test_input_fn = self.setup_predict_graph(features=sentence, positional_features=positions, char_ids=char_ids)
        predict_fn = estimator.predict(input_fn=test_input_fn)

        for predict in predict_fn:
            predictions.append(predict)

        predicted_id_collection = []
        confidence_collection = []

        pred_1_collection = []
        pred_1_confidence_collection = []

        pred_2_collection = []
        pred_2_confidence_collection = []

        pred_3_collection = []
        pred_3_confidence_collection = []

        for each_prediction in predictions:
            predicted_id = []
            confidence = []
            for tag_score in each_prediction["confidence"]:
                confidence.append(tag_score)
            for tag_id in each_prediction["viterbi_seq"]:
                predicted_id.append(self.TAGS_2_ID[tag_id])
            top_3_predicted_indices = each_prediction["top_3_indices"]
            top_3_predicted_confidence = each_prediction["top_3_confidence"]

            # print(top_3_predicted_indices)

            pred_1 = top_3_predicted_indices[:, 0:1].flatten()
            pred_1 = list(map(lambda x: self.TAGS_2_ID[x], pred_1))
            pred_1_collection.append(pred_1)

            pred_2 = top_3_predicted_indices[:, 1:2].flatten()
            pred_2 = list(map(lambda x: self.TAGS_2_ID[x], pred_2))
            pred_2_collection.append(pred_2)

            pred_3 = top_3_predicted_indices[:, 2:].flatten()
            pred_3 = list(map(lambda x: self.TAGS_2_ID[x], pred_3))
            pred_3_collection.append(pred_3)

            pred_1_confidence = top_3_predicted_confidence[:, 0:1]
            pred_2_confidence = top_3_predicted_confidence[:, 1:2]
            pred_3_confidence = top_3_predicted_confidence[:, 2:]

            pred_1_confidence_collection.append(pred_1_confidence)
            pred_2_confidence_collection.append(pred_2_confidence)
            pred_3_confidence_collection.append(pred_3_confidence)

            predicted_id_collection.append(predicted_id)
            confidence_collection.append(confidence)

        return predicted_id_collection, confidence_collection, \
               pred_1_collection, pred_1_confidence_collection, \
               pred_2_collection, pred_2_confidence_collection, \
               pred_3_collection, pred_3_confidence_collection

    @overrides
    def predict_on_test_file(self, estimator, dfs):
        sentences = ["{}".format(SEPERATOR).
                         join(df[self.TEXT_COL].astype(str).values) for df in dfs]

        char_ids = [[[self.char_2_id_map.get(c, 0)
                      for c in word] for word in sentence.split(SEPERATOR)] for sentence in sentences]

        char_ids, char_ids_length = self._pad_sequences(char_ids, pad_tok=int(PAD_CHAR_ID), nlevels=2)

        positions = [df[self.POSITIONAL_COL.split(",")].values.tolist() for df in dfs]

        positions, seq_length = self.pad_position(positions)

        # positions = np.array(positions)

        # TODO add batch support
        predicted_tags_collection, confidence_collection, \
        pred_1_collection, pred_1_confidence_collection, \
        pred_2_collection, pred_2_confidence_collection, \
        pred_3_collection, pred_3_confidence_collection = self.get_tags(estimator=estimator,
                                                                        sentence=sentences,
                                                                        positions=positions,
                                                                        char_ids=char_ids)

        for i, df in enumerate(dfs):
            # print(df.shape, len(predicted_tags_collection[i]))
            # print(df.shape[0])
            # since the batch has variable length sequences, the sequence size is the max sequence length and padded with pad token
            # this was not issue when single file was processed as the single file length was the max seq length
            # Now taking the size from the original sequence and splicing the predicted sequence in order to merge it with the pandas df.
            splice_length = df.shape[0]

            # TODO tidy up this code
            df["predictions"] = predicted_tags_collection[i][:splice_length]
            df["confidence"] = confidence_collection[i][:splice_length]
            df["pred_1"] = pred_1_collection[i][:splice_length]
            df["pred_1_confidence"] = pred_1_confidence_collection[i][:splice_length]
            df["pred_2"] = pred_2_collection[i][:splice_length]
            df["pred_2_confidence"] = pred_2_confidence_collection[i][:splice_length]
            df["pred_3"] = pred_3_collection[i][:splice_length]
            df["pred_3_confidence"] = pred_3_confidence_collection[i][:splice_length]

        return dfs

    @overrides
    def predict_on_test_files(self, estimator,
                              csv_files_path):

        failed_csvs = []
        out_dir = estimator.model_dir + "/predictions/"
        check_n_makedirs(out_dir)

        files = [file for file in os.listdir(csv_files_path) if file.endswith('.csv')]
        batchsize = 12
        index = 0
        remaining = len(files)
        progress_bar = tqdm(total=len(files))

        while remaining > 0:
            batch = min(remaining, batchsize)

            print('NEW BATCH\n')
            dfs = []

            for csv_file in files[index:index + batch]:
                df = pd.read_csv(os.path.join(csv_files_path, csv_file)).fillna(UNKNOWN_WORD)
                df.file_name = csv_file
                dfs.append(df)

            dfs = self.predict_on_test_file(estimator, dfs)

            for predicted_df in dfs:
                print_info(predicted_df.file_name)
                predicted_df.to_csv(out_dir + ntpath.basename(predicted_df.file_name), index=False)

            index += batch
            remaining -= batch
            progress_bar.update(index)

        progress_bar.close()
        return out_dir

    @overrides
    def predict_on_text(self, estimator, sentence):

        # Convert space delimited text to a sentence delimited  by `SEPERATOR`
        sentence = sentence.split()
        sentence = "{}".format(SEPERATOR).join(sentence)

        # Trailing by 213, Somerset got a solid start to their second innings before Simmons stepped in to bundle them out for 174.
        char_ids = [[self.char_2_id_map.get(c, 0) for c in word] for word in sentence.split(SEPERATOR)]
        char_ids, char_ids_length = self._pad_sequences([char_ids], pad_tok=0, nlevels=2)

        predicted_tags, confidence, pred_1, pred_1_confidence, pred_2, pred_2_confidence, \
        pred_3, pred_3_confidence = self.get_tags(estimator=estimator, sentence=sentence, positions=[0, 0, 0],
                                                  char_ids=char_ids)
        # TODO fix the positions for the single text example
        print_info(predicted_tags)
        return predicted_tags
