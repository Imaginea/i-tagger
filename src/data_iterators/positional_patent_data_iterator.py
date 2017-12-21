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
from tensorflow.python.platform import gfile
from interfaces.two_features_interface import IPostionalFeature


class PositionalPatentIDataIterator(IDataIterator, IPostionalFeature):
    def __init__(self, data_dir, batch_size):
        IDataIterator.__init__(self, data_dir, batch_size)
        IPostionalFeature.__init__(self)
        self.config = ConfigManager("src/config/patent_data_preprocessor.ini")

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

    def _pad_sequences(self, sequences, pad_tok, nlevels, MAX_WORD_LENGTH=20):
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
            max_length = max(map(lambda x: len(x.split(" ")), sequences))
            # sequence_padded, sequence_length = _pad_sequences(sequences,
            #                                                   pad_tok, max_length)
            # breaking the code to pad the string instead on its ids
            for seq in sequences:
                current_length = len(seq.split(" "))
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

        return sequence_padded, sequence_length

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
        return sequence_padded, max_length

    def _make_seq_pair(self, text_file_path, char_2_id_map, use_char_embd):
        '''
        Reads the CoNLL text file and makes Sentence-Tags pair for NN model
        :param text_file_path:
        :param word_col:
        :param tag_col:
        :param empty_line_filler:
        :return:
        '''

        self.TEXT_COL = self.config.get_item("Schema", "text_column")
        self.ENTITY_COL = self.config.get_item("Schema", "entity_column")
        self.POSITIONAL_COL = self.config.get_item("Schema", "positional_column")

        df = pd.read_csv(text_file_path,
                         delimiter=SEPRATOR,
                         skip_blank_lines=False,
                         quotechar=QUOTECHAR).fillna(EMPTY_LINE_FILLER)

        # get the column values
        sequences = df[self.TEXT_COL].values
        labels = df[self.ENTITY_COL].values
        positions = df[self.POSITIONAL_COL.split(",")].values.tolist()

        list_text = []
        list_postions = []
        list_char_ids = []
        list_tag = []

        # [feature1 ,feature2, label]
        sentence_feature1 = []
        char_ids_feature2 = []
        positional_feature3 = []
        tag_label = []
        for word, position, tag in tqdm(zip(sequences, positions, labels)):
            if word != EMPTY_LINE_FILLER:  # collect the sequence data till new line
                list_text.append(word)
                list_postions.append(position)
                try:
                    word_2_char_ids = [char_2_id_map.get(c, 0) for c in word]
                    list_char_ids.append(word_2_char_ids)
                except Exception as e:
                    print("Exception: ", e.with_traceback())
                    print(word)  # quotechar="^" added after debugging when " created problem
                    exit()
                list_tag.append(tag)
            else:  # when a new line encountered, make an example with feature and label
                sentence_feature1.append(" ".join(list_text))
                char_ids_feature2.append(list_char_ids)
                positional_feature3.append(list_postions)
                tag_label.append(" ".join(list_tag))

                # Make the container empty
                list_text = []
                list_char_ids = []
                list_postions = []
                list_tag = []

        # pdb.set_trace()
        # length = len(sorted(char_ids_feature2,key=len, reverse=True)[0])
        # print_error(length)
        # char_ids_feature2 = np.array([np.array(xi+[0]*(length-len(xi))) for xi in char_ids_feature2])

        positional_feature3, seq_length = self.pad_position(positional_feature3)

        positional_feature3 = np.array(positional_feature3)

        if use_char_embd:
            sentence_feature1, seq_length = self._pad_sequences(sentence_feature1, nlevels=1,
                                                                pad_tok=" <PAD>")  # space is used so that it can append to the string sequence
            sentence_feature1 = np.array(sentence_feature1)

            char_ids_feature2, seq_length = self._pad_sequences(char_ids_feature2, nlevels=2, pad_tok=0)
            char_ids_feature2 = np.array(char_ids_feature2)
            seq_length = np.array(seq_length)
            # print_warn(seq_length.shape)
            # exit()
            tag_label, seq_length = self._pad_sequences(tag_label, nlevels=1, pad_tok=" <PAD>")
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
                    char_ids_placeholder = tf.placeholder(tf.int32, [None, None, 20], name="char_ids")
                labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name="label")

                # Build dataset iterator
                if use_char_embd:
                    dataset = tf.data.Dataset.from_tensor_slices(({"text": text_features_placeholder,
                                                                   "char_ids": char_ids_placeholder,
                                                                   "position": positional_features_placeholder},
                                                                  labels_placeholder))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices(({"text": text_features_placeholder,
                                                                   "position": positional_features_placeholder},
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

    def predict_inputs(self, features, positional_features, char_ids, batch_size=1, scope='test-data'):
        """Returns test set as Operations.
        Returns:
            (features, ) Operations that iterate over the test set.
        """
        # Convert raw sentence into a lisr, since TF works on only list/matrix
        if not isinstance(features, list):
            features = [features]

        if not isinstance(positional_features, list):
            positional_features = [positional_features]
            positional_features = np.asarray(positional_features)

        # TODO mages why this is not below doc in scope below
        positional_features = tf.constant(positional_features, dtype=tf.float32)

        def inputs():
            with tf.name_scope(scope):
                docs = tf.constant(features, dtype=tf.string)

                dataset = tf.data.Dataset.from_tensor_slices(({"text": docs,
                                                               "position": positional_features,
                                                               "char_ids": char_ids},))
                dataset.repeat(1)
                # Return as iteration in batches of 1
                return dataset.batch(batch_size).make_one_shot_iterator().get_next()

        return inputs

    @overrides
    def setup_train_input_graph(self):
        train_sentences, train_char_ids, train_positions, train_ner_tags = \
            self._make_seq_pair(text_file_path=self.preprocessed_data_info.TRAIN_DATA_FILE,
                                char_2_id_map=self.preprocessed_data_info.char_2_id_map,
                                use_char_embd=True)  # TODO
        self.NUM_TRAINING_SAMPLES = train_sentences.shape[0]  # TODO

        self.train_data_input_fn, self.train_data_init_hook = self._setup_input_graph2(text_features=train_sentences,
                                                                                       positional_features=train_positions,
                                                                                       char_ids=train_char_ids,
                                                                                       labels=train_ner_tags,
                                                                                       batch_size=self.BATCH_SIZE,
                                                                                       use_char_embd=True)  # TODO

    @overrides
    def setup_val_input_graph(self):
        val_sentences, val_char_ids, eval_positions, val_ner_tags = \
            self._make_seq_pair(text_file_path=self.preprocessed_data_info.VAL_DATA_FILE,
                                char_2_id_map=self.preprocessed_data_info.char_2_id_map,
                                use_char_embd=True)  # TODO
        self.val_data_input_fn, self.val_data_init_hook = self._setup_input_graph2(text_features=val_sentences,
                                                                                   positional_features=eval_positions,
                                                                                   char_ids=val_char_ids,
                                                                                   labels=val_ner_tags,
                                                                                   batch_size=self.BATCH_SIZE,
                                                                                   use_char_embd=True,
                                                                                   is_eval=True)  # TODO

    # @overrides
    # def setup_predict_input_graph(self):
    #     #TODO this is not used, since we need to append the predicted value to the CSV files
    #     test_sentences,test_char_ids, test_ner_tags = \
    #         self._make_seq_pair(text_file_path=self.preprocessed_data_info.TEST_DATA_FILE,
    #                             char_2_id_map=self.preprocessed_data_info.char_2_id_map,
    #                             use_char_embd=True)
    #
    #     self.val_data_input_fn, self.val_data_init_hook = self._setup_input_graph2(text_features=test_sentences,
    #                                                                                  char_ids=test_char_ids,
    #                                                                                  labels=test_ner_tags,
    #                                                                                  batch_size=self.BATCH_SIZE,
    #                                                                                  use_char_embd=True) #TODO

    def get_tags(self, estimator, sentence, positions, char_ids, tag_vocab_tsv):

        with gfile.Open(tag_vocab_tsv, 'r') as file:
            ner_vocab = list(map(lambda x: x.strip(), file.readlines()))
            tags_vocab = {id_num: tag for id_num, tag in enumerate(ner_vocab)}

        predictions = []
        test_input_fn = self.predict_inputs(features=sentence, positional_features=positions, char_ids=char_ids)
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

    def predict_on_csv_files(self, estimator,
                             csv_files_path):

        positional_columns = self.config.get_item("Schema", "positional_column")

        for csv_file in tqdm(os.listdir(csv_files_path)):
            sentence = ""
            csv_file = os.path.join(csv_files_path, csv_file)
            if csv_file.endswith(".csv"):
                print_info(csv_file)
                # print_info("processing ====> {}".format(csv_file))
                df = pd.read_csv(csv_file).fillna(UNKNOWN_WORD)
                # df = io_2_iob(df, entity_col, entity_iob_col) # removing since we are using preprocessed test folder TODO chain IOB
                sentence = (" ".join(df[self.preprocessed_data_info.TEXT_COL].values))

                positions = df[positional_columns.split(",")].astype(float).values

                char_ids = [[self.preprocessed_data_info.char_2_id_map.get(c, 0) for c in word] for word in
                            sentence.split(" ")]
                char_ids, char_ids_length = self._pad_sequences([char_ids], pad_tok=0, nlevels=2)

                # TODO add batch support
                predicted_tags, confidence, pred_1, pred_1_confidence, pred_2, pred_2_confidence, \
                pred_3, pred_3_confidence = self.get_tags(estimator, sentence, positions, char_ids,
                                                          self.preprocessed_data_info.ENTITY_VOCAB_FILE)
                df["predictions"] = predicted_tags
                df["confidence"] = confidence
                df["pred_1"] = pred_1
                df["pred_1_confidence"] = pred_1_confidence
                df["pred_2"] = pred_2
                df["pred_2_confidence"] = pred_2_confidence
                df["pred_3"] = pred_3
                df["pred_3_confidence"] = pred_3_confidence

                out_dir = estimator.model_dir + "/predictions/"
                check_n_makedirs(out_dir)
                df.to_csv(out_dir + ntpath.basename(csv_file), index=False)

        return None
