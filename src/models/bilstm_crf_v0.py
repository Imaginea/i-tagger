# https://github.com/guillaumegenthial/sequence_tagging
# https://github.com/jiaqianghuai/tf-lstm-crf-batch
# https://www.tensorflow.org/api_docs/python/tf/contrib/crf
# https://github.com/Franck-Dernoncourt/NeuroNER
# https://www.clips.uantwerpen.be/conll2003/ner/
# https://stackoverflow.com/questions/3330227/free-tagged-corpus-for-named-entity-recognition

# https://sites.google.com/site/ermasoftware/getting-started/ne-tagging-conll2003-data
# Dataset: https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003
# Reference: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/estimators/abalone.py
# https://github.com/tensorflow/tensorflow/issues/14018

from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys

from config.global_constants import *
from config.preprocessed_data_info import PreprocessedDataInfo
from helpers.os_helper import check_n_makedirs
from helpers.print_helper import *
from helpers.tf_data_helper import *
# from interfaces.data_iterator import
from helpers.tf_hooks.pre_run import PreRunTaskHook
from interfaces.model_configs import IModelConfig
from interfaces.two_features_interface import ITextFeature

tf.logging.set_verbosity("INFO")


class BiLSTMCRFConfigV0(IModelConfig):
    def __init__(self,
                 model_dir,
                 vocab_size,
                 char_vocab_size,
                 number_tags,
                 unknown_word,
                 pad_word,
                 tags_vocab_file,
                 words_vocab_file,
                 chars_vocab_file,
                 # hyper parameters
                 use_char_embedding,
                 learning_rate,
                 word_level_lstm_hidden_size,
                 char_level_lstm_hidden_size,
                 word_emd_size,
                 char_emd_size,
                 num_lstm_layers,
                 out_keep_propability,
                 use_crf):

        # Constant params
        self.MODEL_DIR = model_dir
        self.UNKNOWN_WORD = unknown_word
        self.PAD_WORD = pad_word
        self.UNKNOWN_TAG = "O"  # TODO

        # Preprocessing Paramaters
        self.TAGS_VOCAB_FILE = tags_vocab_file
        self.WORDS_VOCAB_FILE = words_vocab_file
        self.CHARS_VOCAB_FILE = chars_vocab_file

        self.VOCAB_SIZE = vocab_size
        self.CHAR_VOCAB_SIZE = char_vocab_size
        self.NUM_TAGS = number_tags

        # Model hyper parameters
        self.USE_CRF = use_crf
        self.USE_CHAR_EMBEDDING = use_char_embedding
        self.LEARNING_RATE = learning_rate
        self.KEEP_PROP = out_keep_propability
        self.WORD_EMBEDDING_SIZE = word_emd_size
        self.CHAR_EMBEDDING_SIZE = char_emd_size
        self.WORD_LEVEL_LSTM_HIDDEN_SIZE = word_level_lstm_hidden_size
        self.CHAR_LEVEL_LSTM_HIDDEN_SIZE = char_level_lstm_hidden_size
        self.NUM_LSTM_LAYERS = num_lstm_layers

    @staticmethod
    def with_user_hyperparamaters(experiment_root_dir, data_iterator):

        # preprocessed_data_info = PreprocessedDataInfo.load(experiment_root_dir)

        use_crf = "y"  # TODO
        use_char_embedding = False
        char_level_lstm_hidden_size = 32  # default
        char_emd_size = 32  # default

        if use_crf == 'y':
            use_crf = True
        else:
            use_crf = False

        use_char_embedding_option = input("use_char_embedding (y/n): ") or "y"
        learning_rate = input("learning_rate (0.001): ") or 0.001
        learning_rate = float(learning_rate)
        num_lstm_layers = input("num_word_lstm_layers (2): ") or 2
        num_lstm_layers = int(num_lstm_layers)

        if use_char_embedding_option == 'y':
            use_char_embedding = True
            char_level_lstm_hidden_size = input("char_level_lstm_hidden_size (48): ") or 48
            char_level_lstm_hidden_size = int(char_level_lstm_hidden_size)
            char_emd_size = input("char_emd_size (48): ") or 48
            char_emd_size = int(char_emd_size)
        else:
            use_char_embedding = False

        word_level_lstm_hidden_size = input("word_level_lstm_hidden_size (64): ") or 64
        word_level_lstm_hidden_size = int(word_level_lstm_hidden_size)
        word_emd_size = input("word_emd_size (64): ") or 64
        word_emd_size = int(word_emd_size)
        out_keep_propability = input("out_keep_propability(0.5) : ") or 0.5
        out_keep_propability = float(out_keep_propability)

        # Does this sound logical? review please
        '''
        experiment_root_dir/
            - data_iterator/
                - model_name/
                    - user_hyper_params/
        '''
        model_dir = experiment_root_dir + "/" + data_iterator.NAME + "/bilstm_crf_v0/" + \
                    "charembd_{}_lr_{}_lstmsize_{}-{}-{}_wemb_{}_cemb_{}_outprob_{}".format(
                        str(use_char_embedding),
                        learning_rate,
                        num_lstm_layers,
                        word_level_lstm_hidden_size,
                        char_level_lstm_hidden_size,
                        word_emd_size,
                        char_emd_size,
                        out_keep_propability)

        model_config = BiLSTMCRFConfigV0(model_dir=model_dir,
                                         vocab_size=data_iterator.VOCAB_SIZE,
                                         char_vocab_size=data_iterator.CHAR_VOCAB_SIZE,
                                         number_tags=data_iterator.NUM_TAGS,
                                         unknown_word=UNKNOWN_WORD,
                                         pad_word=PAD_WORD,
                                         tags_vocab_file=data_iterator.ENTITY_VOCAB_FILE,
                                         words_vocab_file=data_iterator.WORDS_VOCAB_FILE,
                                         chars_vocab_file=data_iterator.CHAR_VOCAB_SIZE,
                                         # hyper parameters
                                         use_char_embedding=use_char_embedding,
                                         learning_rate=learning_rate,
                                         word_level_lstm_hidden_size=word_level_lstm_hidden_size,
                                         char_level_lstm_hidden_size=char_level_lstm_hidden_size,
                                         word_emd_size=word_emd_size,
                                         char_emd_size=char_emd_size,
                                         num_lstm_layers=num_lstm_layers,
                                         out_keep_propability=out_keep_propability,
                                         use_crf=True)
        check_n_makedirs(model_dir)
        IModelConfig.save(model_dir=model_dir, config=model_config)

        return model_config


# =======================================================================================================================


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
# run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
run_config.allow_soft_placement = True
run_config.log_device_placement = False
run_config = tf.contrib.learn.RunConfig(session_config=run_config,
                                        save_checkpoints_steps=50,
                                        keep_checkpoint_max=3,
                                        save_summary_steps=50)


class BiLSTMCRFV0(tf.estimator.Estimator, ITextFeature):
    def __init__(self,
                 ner_config: BiLSTMCRFConfigV0):
        tf.estimator.Estimator.__init__(self,
                                        model_fn=self._model_fn,
                                        model_dir=ner_config.MODEL_DIR,
                                        config=run_config)

        ITextFeature.__init__(self)

        self.ner_config = ner_config

        self.hooks = []

    def _model_fn(self, features, labels, mode, params):
        '''

        :param features: TF Placeholder of type dict of shape [BATCH_SIZE, 1]
        :param labels: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param mode: ModeKeys
        :param params:
        :return:
        '''

        is_training = mode == ModeKeys.TRAIN

        # [BATCH_SIZE, 1]
        text_features = features[self.FEATURE_1_NAME]

        if self.ner_config.USE_CHAR_EMBEDDING:
            # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH]
            char_ids = features[self.FEATURE_2_NAME]

            tf.logging.info('char_ids: =======> {}'.format(char_ids))

            s = tf.shape(char_ids)

            # remove pad words
            char_ids_reshaped = tf.reshape(char_ids, shape=(s[0] * s[1], s[2]))  # 20 -> char dim

        with tf.variable_scope("sentence-words-2-ids"):
            word_table = lookup.index_table_from_file(vocabulary_file=self.ner_config.WORDS_VOCAB_FILE,
                                                      num_oov_buckets=0,  # TODO use this for Out of Vocab
                                                      default_value=1,  # id of <UNK>  w.r.t WORD VOCAB
                                                      name="table")
            tf.logging.info('table info: {}'.format(word_table))

            # [BATCH_SIZE, 1]
            words = tf.string_split(text_features, delimiter=SEPERATOR)

            # [BATCH_SIZE, ?] i.e [BATCH_SIZE, VARIABLE_SEQ_LENGTH]
            densewords = tf.sparse_tensor_to_dense(words,
                                                   default_value=self.ner_config.PAD_WORD)  # TODO add test case

            # [BATCH_SIZE, ?] i.e [BATCH_SIZE, MAX_SEQ_LENGTH]
            token_ids = word_table.lookup(densewords)  # TODO check is it variable length or not?

        with tf.variable_scope("ner-tags-2-ids"):
            if mode != ModeKeys.INFER:
                ner_table = lookup.index_table_from_file(vocabulary_file=self.ner_config.TAGS_VOCAB_FILE,
                                                         num_oov_buckets=0,
                                                         default_value=0,  # id of <UNK> w.r.t ENTITY VOCAB
                                                         name="table")

                tf.logging.info('ner_table info: {}'.format(ner_table))

                # [BATCH_SIZE, 1]
                labels_splitted = tf.string_split(labels, delimiter=SEPERATOR)
                # [BATCH_SIZE, ?] i.e [BATCH_SIZE, VARIABLE_SEQ_LENGTH]
                labels_splitted_dense = tf.sparse_tensor_to_dense(labels_splitted,
                                                                  default_value="O")
                # [BATCH_SIZE, ?] i.e [BATCH_SIZE, MAX_SEQ_LENGTH]
                ner_ids = ner_table.lookup(labels_splitted_dense)
                ner_ids = tf.cast(ner_ids, tf.int32)

                tf.logging.info("ner_ids: {}".format(ner_ids))

        with tf.variable_scope("word-embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [BATCH_SIZE, MAX_SEQ_LENGTH] --->  [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE].
            word_embeddings = tf.contrib.layers.embed_sequence(token_ids,
                                                               vocab_size=self.ner_config.VOCAB_SIZE,
                                                               embed_dim=self.ner_config.WORD_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            word_embeddings = tf.layers.dropout(word_embeddings,
                                                rate=self.ner_config.KEEP_PROP,
                                                seed=42,
                                                training=mode == tf.estimator.ModeKeys.TRAIN)

            # [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE]
            tf.logging.info('word_embeddings =====> {}'.format(word_embeddings))

            # seq_length = get_sequence_length_old(word_embeddings) TODO working
            # [BATCH_SIZE, ]
            seq_length = get_sequence_length(token_ids)

            tf.logging.info('seq_length =====> {}'.format(seq_length))

        with tf.variable_scope("char_embed_layer"):
            if self.ner_config.USE_CHAR_EMBEDDING:
                print_error((self.ner_config.CHAR_VOCAB_SIZE, self.ner_config.CHAR_EMBEDDING_SIZE))
                char_embeddings = tf.contrib.layers.embed_sequence(char_ids,
                                                                   vocab_size=self.ner_config.CHAR_VOCAB_SIZE,
                                                                   embed_dim=self.ner_config.CHAR_EMBEDDING_SIZE,
                                                                   initializer=tf.contrib.layers.xavier_initializer(
                                                                       seed=42))

                # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
                char_embeddings = tf.layers.dropout(char_embeddings,
                                                    rate=self.ner_config.KEEP_PROP,
                                                    seed=42,
                                                    training=mode == tf.estimator.ModeKeys.TRAIN)  # TODO add test case

                tf.logging.info('char_embeddings =====> {}'.format(char_embeddings))

        with tf.variable_scope("chars_level_bilstm_layer"):
            if self.ner_config.USE_CHAR_EMBEDDING:
                # put the time dimension on axis=1
                shape = tf.shape(char_embeddings)

                BATCH_SIZE = shape[0]
                MAX_DOC_LENGTH = shape[1]
                CHAR_MAX_LENGTH = shape[2]

                TOTAL_DOCS_LENGTH = tf.reduce_sum(seq_length)

                # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]  ===>
                #      [BATCH_SIZE * MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[BATCH_SIZE * MAX_DOC_LENGTH, CHAR_MAX_LENGTH,
                                                    self.ner_config.CHAR_EMBEDDING_SIZE],
                                             name="reduce_dimension_1")

                tf.logging.info('reshaped char_embeddings =====> {}'.format(char_embeddings))

                # word_lengths = get_sequence_length_old(char_embeddings) TODO working
                word_lengths = get_sequence_length(char_ids_reshaped)

                tf.logging.info('word_lengths =====> {}'.format(word_lengths))

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.ner_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.ner_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                                  state_is_tuple=True)

                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    dtype=tf.float32,
                    sequence_length=word_lengths,
                    inputs=char_embeddings,
                    scope="encode_words")

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                encoded_words = tf.concat([output_fw, output_bw], axis=-1)

                # [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE]
                encoded_words = tf.reshape(encoded_words,
                                           shape=[BATCH_SIZE, MAX_DOC_LENGTH, 2 *
                                                  self.ner_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE])

                tf.logging.info('encoded_words =====> {}'.format(encoded_words))

        with  tf.variable_scope("word_level_lstm_layer"):
            # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
            d_rnn_cell_fw_one = tf.nn.rnn_cell.LSTMCell(self.ner_config.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.LSTMCell(self.ner_config.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)

            if is_training:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one,
                                                                  output_keep_prob=self.ner_config.KEEP_PROP)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one,
                                                                  output_keep_prob=self.ner_config.KEEP_PROP)
            else:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one, output_keep_prob=1.0)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one, output_keep_prob=1.0)

            d_rnn_cell_fw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_fw_one] *
                                                                  self.ner_config.NUM_LSTM_LAYERS,
                                                            state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_bw_one] *
                                                                  self.ner_config.NUM_LSTM_LAYERS,
                                                            state_is_tuple=True)

            (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=d_rnn_cell_fw_one,
                cell_bw=d_rnn_cell_bw_one,
                dtype=tf.float32,
                sequence_length=seq_length,
                inputs=word_embeddings,
                scope="encod_sentence")

            # [BATCH_SIZE, MAX_SEQ_LENGTH, 2*WORD_LEVEL_LSTM_HIDDEN_SIZE) TODO check MAX_SEQ_LENGTH?
            encoded_sentence = tf.concat([fw_output_one,
                                          bw_output_one], axis=-1)

            tf.logging.info('encoded_sentence =====> {}'.format(encoded_sentence))

        with tf.variable_scope("char_word_embeddings-mergeing_layer"):
            if self.ner_config.USE_CHAR_EMBEDDING:
                encoded_doc = tf.concat([encoded_words, encoded_sentence], axis=-1, name="sentence_words_concat")
            else:
                encoded_doc = encoded_sentence

            # [BATCH_SIZE, MAX_SEQ_LENGTH, 2*WORD_LEVEL_LSTM_HIDDEN_SIZE + 2*CHAR_LEVEL_LSTM_HIDDEN_SIZE]
            encoded_doc = tf.layers.dropout(encoded_doc,
                                            rate=self.ner_config.KEEP_PROP,
                                            seed=42,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)

            tf.logging.info('encoded_doc: =====> {}'.format(encoded_doc))

        with tf.variable_scope("projection"):

            NUM_WORD_LSTM_NETWORKS = 1 + 1  # word_level_lstm_layer BiDirectional
            NUM_CHAR_LSTM_NETWORKS = 1 + 1  # char_level_lstm_layer BiDirectional

            # Example: If WORD_LEVEL_LSTM_HIDDEN_SIZE = 300, CHAR_LEVEL_LSTM_HIDDEN_SIZE = 300,
            # NEW_SHAPE = 2 * 300 + 2 * 300 = 1200
            NEW_SHAPE = NUM_WORD_LSTM_NETWORKS * self.ner_config.WORD_LEVEL_LSTM_HIDDEN_SIZE + \
                        NUM_CHAR_LSTM_NETWORKS * self.ner_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE

            if self.ner_config.USE_CHAR_EMBEDDING:
                # [NEW_SHAPE, NUM_TAGS]
                W = tf.get_variable("W", dtype=tf.float32,
                                    shape=[NEW_SHAPE, self.ner_config.NUM_TAGS])
                # [NUM_TAGS]
                b = tf.get_variable("b", shape=[self.ner_config.NUM_TAGS],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                # [NUM_WORD_LSTM_NETWORKS * WORD_LEVEL_LSTM_HIDDEN_SIZE, NUM_TAGS]
                W = tf.get_variable("W", dtype=tf.float32,
                                    shape=[NUM_WORD_LSTM_NETWORKS * self.ner_config.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                           self.ner_config.NUM_TAGS])
                # [NUM_TAGS]
                b = tf.get_variable("b", shape=[self.ner_config.NUM_TAGS],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
            # [MAX_SEQ_LENGTH]
            nsteps = tf.shape(encoded_doc)[1]

            tf.logging.info("nsteps: =====> {} ".format(nsteps))

            if self.ner_config.USE_CHAR_EMBEDDING:
                encoded_doc = tf.reshape(encoded_doc, [-1, NEW_SHAPE],
                                         name="reshape_encoded_doc")
            else:
                encoded_doc = tf.reshape(encoded_doc,
                                         [-1, NUM_WORD_LSTM_NETWORKS * self.ner_config.WORD_LEVEL_LSTM_HIDDEN_SIZE],
                                         name="reshape_encoded_doc")

            tf.logging.info("encoded_doc: {}".format(encoded_doc))
            encoded_doc = tf.matmul(encoded_doc, W) + b

            tf.logging.info("encoded_doc: {}".format(encoded_doc))
            # [BATCH_SIZE, MAX_SEQ_LENGTH, NUM_TAGS]
            logits = tf.reshape(encoded_doc, [-1, nsteps, self.ner_config.NUM_TAGS], name="reshape_predictions")
            tf.logging.info("logits: {}".format(logits))

        with  tf.variable_scope("loss-layer"):
            """Defines the loss"""

            if mode == ModeKeys.INFER:
                ner_ids = tf.placeholder(tf.int32, shape=[None, None],
                                         name="labels")  # no labels during prediction
            else:
                ner_ids = ner_ids

            if True:  # self.config.use_crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logits, ner_ids, seq_length)

                tf.logging.info("log_likelihood:  =====> {}".format(log_likelihood))

                # [NUM_TAGS, NUM_TAGS]
                trans_params = trans_params  # need to evaluate it for decoding
                tf.logging.info("trans_params: =====> {}".format(trans_params))
                ner_crf_loss = tf.reduce_mean(-log_likelihood)

                tf.summary.scalar("loss", ner_crf_loss)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=ner_ids)
                mask = tf.sequence_mask(seq_length)
                losses = tf.boolean_mask(losses, mask)
                ner_crf_loss = tf.reduce_mean(losses)
                tf.summary.scalar("loss", ner_crf_loss)

            viterbi_seq, best_score = tf.contrib.crf.crf_decode(logits, trans_params, seq_length)

            tf.logging.info("viterbi_seq: {}".format(viterbi_seq))

            predictions = { #TODO features class
                "classes": tf.cast(tf.argmax(logits, axis=-1),
                                   tf.int32),
                # [BATCH_SIZE, SEQ_LEN]
                "viterbi_seq": viterbi_seq,
                # [BATCH_SIZE]
                "confidence": tf.reduce_max(tf.nn.softmax(logits, dim=-1), axis=-1),

                "top_3_indices": tf.nn.top_k(tf.nn.softmax(logits, dim=-1), k=3).indices,

                "top_3_confidence": tf.nn.top_k(tf.nn.softmax(logits, dim=-1), k=3).values
            }

        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            train_op = tf.contrib.layers.optimize_loss(
                loss=ner_crf_loss,
                global_step=tf.train.get_global_step(),
                optimizer=tf.train.AdamOptimizer,
                learning_rate=self.ner_config.LEARNING_RATE)

            loss = ner_crf_loss

            eval_metric_ops = {
                'Accuracy': tf.metrics.accuracy(
                    labels=ner_ids,
                    predictions=predictions["viterbi_seq"],
                    name='accuracy'),
                'Precision': tf.metrics.precision(
                    labels=ner_ids,
                    predictions=predictions["viterbi_seq"],
                    name='Precision'),
                'Recall': tf.metrics.recall(
                    labels=ner_ids,
                    predictions=predictions["viterbi_seq"],
                    name='Recall')
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            # training_hooks=self.hooks
        )

    def set_shape_hook(self, tensor_name):
        print_info("adding " + tensor_name)

        def save_embed_mat(sess):
            graph = sess.graph
            tensor = graph.get_tensor_by_name(tensor_name)

            shape = sess.run(tf.shape(tensor))

            print_error(" ============ \n {} shape is {} \n=============\n".format(tensor, shape))

        hook = PreRunTaskHook()
        hook.user_func = save_embed_mat

        self.hooks.append(hook)

    def get_shape_hooks(self):
        self.set_shape_hook("char_embed_layer/dropout/dropout/mul:0")
        return self.hooks
