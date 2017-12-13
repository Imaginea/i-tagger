import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import tensorflow as tf
from interfaces.data_iterator import DataIterator
from helpers.tf_hooks.data_initializers import DataIteratorInitializerHook

class PatentDataIterator(DataIterator):
    def __init__(self, data_dir,  batch_size):
        super(PatentDataIterator, self).__init__(data_dir,  batch_size)

    def _pad_sequences(self, sequences, pad_tok, max_length):
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

    def pad_sequences(self, sequences, pad_tok, nlevels, MAX_WORD_LENGTH=20):
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
        elif nlevels == 2:
            # max_length_word = max([max(map(lambda x: len(x), seq))
            #                        for seq in sequences])
            sequence_padded, sequence_length = [], []
            for seq in tqdm(sequences):
                # all words are same length now
                sp, sl = self._pad_sequences(seq, pad_tok, MAX_WORD_LENGTH)
                sequence_padded += [sp]
                sequence_length += [sl]

            max_length_sentence = max(map(lambda x: len(x), sequences))
            sequence_padded, _ = self._pad_sequences(sequence_padded,
                                                     [pad_tok] * MAX_WORD_LENGTH,
                                                     max_length_sentence)  # TODO revert -1 to pad_tok
            sequence_length, _ = self._pad_sequences(sequence_length, 0,
                                                     max_length_sentence)

        return sequence_padded, sequence_length

    def _make_seq_pair(self, text_file_path, char_2_id_map, use_char_embd):
        '''
        Reads the CoNLL text file and makes Sentence-Tags pair for NN model
        :param text_file_path:
        :param word_col:
        :param tag_col:
        :param empty_line_filler:
        :return:
        '''

        word_col = "word"
        tag_col = "entity_name"
        empty_line_filler = "<LINE_END>"

        df = pd.read_csv(text_file_path,
                         delimiter=" ",
                         header=None,
                         skip_blank_lines=False,
                         quotechar="^").fillna(empty_line_filler)

        columns = [word_col, tag_col, "doc_id"]  # define columns #TODO 1
        df.columns = columns

        # get the column values
        sequences = df[word_col].values
        labels = df[tag_col].values

        list_text = []
        list_char_ids = []
        list_tag = []

        # [feature1 ,feature2, label]
        sentence_feature1 = []
        char_ids_feature2 = []
        tag_label = []

        for word, tag in tqdm(zip(sequences, labels)):
            if word != empty_line_filler:  # collect the sequence data till new line
                list_text.append(word)
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
                tag_label.append(" ".join(list_tag))

                # Make the container empty
                list_text = []
                list_char_ids = []
                list_tag = []

        # pdb.set_trace()
        # length = len(sorted(char_ids_feature2,key=len, reverse=True)[0])
        # print_error(length)
        # char_ids_feature2 = np.array([np.array(xi+[0]*(length-len(xi))) for xi in char_ids_feature2])


        # pdb.set_trace()

        if use_char_embd:
            sentence_feature1, seq_length = self.pad_sequences(sentence_feature1, nlevels=1,
                                                               pad_tok=" <PAD>")  # space is used so that it can append to the string sequence
            sentence_feature1 = np.array(sentence_feature1)

            char_ids_feature2, seq_length = self.pad_sequences(char_ids_feature2, nlevels=2, pad_tok=0)
            char_ids_feature2 = np.array(char_ids_feature2)
            seq_length = np.array(seq_length)
            # print_warn(seq_length.shape)
            # exit()
            tag_label, seq_length = self.pad_sequences(tag_label, nlevels=1, pad_tok=" <PAD>")
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
        # tf.logging.info("numeric_features.shape: =====> {}".format(char_ids.shape))
        tf.logging.info("labels.shape: =====> {}".format(labels.shape))

        tf.logging.info("text_features.shape: =====> {}".format(type(text_features)))
        tf.logging.info("numeric_features.shape: =====> {}".format(type(char_ids)))
        char_ids = np.array(char_ids)
        tf.logging.info("labels.shape: =====> {}".format(type(labels)))

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
                if use_char_embd:
                    char_ids_placeholder = tf.placeholder(tf.int32, [None, None, 20], name="char_ids")
                labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name="label")

                # Build dataset iterator
                if use_char_embd:
                    dataset = tf.data.Dataset.from_tensor_slices(({"text": text_features_placeholder,
                                                                   "char_ids": char_ids_placeholder},
                                                                  labels_placeholder))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices(({"text": text_features_placeholder},
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

    def setup_train_input_graph(self):
        train_sentences, train_char_ids, train_ner_tags = \
            self._make_seq_pair(text_file_path=self.preprocessed_data_info.TRAIN_DATA_FILE,
                                char_2_id_map=self.preprocessed_data_info.char_2_id_map,
                                use_char_embd=self.preprocessed_data_info.USE_CHAR_EMBEDDING)
        self.train_data_inputs, self.train_data_init_hook = self._setup_input_graph2(text_features=train_sentences,
                                                                                     char_ids=train_char_ids,
                                                                                     labels=train_ner_tags,
                                                                                     batch_size=self.BATCH_SIZE,
                                                                                     use_char_embd=self.preprocessed_data_info.USE_CHAR_EMBEDDING) #TODO

    def setup_val_input_graph(self):
        val_sentences, val_char_ids, val_ner_tags = \
            self._make_seq_pair(text_file_path=self.preprocessed_data_info.VAL_DATA_FILE,
                                char_2_id_map=self.preprocessed_data_info.char_2_id_map,
                                use_char_embd=self.preprocessed_data_info.USE_CHAR_EMBEDDING)
        self.train_data_inputs, self.train_data_init_hook = self._setup_input_graph2(text_features=val_sentences,
                                                                                     char_ids=val_char_ids,
                                                                                     labels=val_ner_tags,
                                                                                     batch_size=self.BATCH_SIZE,
                                                                                     use_char_embd=self.preprocessed_data_info.USE_CHAR_EMBEDDING) #TODO

    def setup_predict_input_graph(self):
        raise NotImplementedError
