# import tensorflow as tf
#
# def get_global_config():
#
#     flags = tf.app.flags
#
#     ####################################
#     #  PatentDataPreprocessor Config   #
#     ###################################
#
#     flags.DEFINE_string("over_write","no","")
#     flags.DEFINE_string("use_iob_format","yes","(yes/no) enable this option if your tags needs to be formatted with IOB format")
#
#     # Schema
#     flags.DEFINE_string("text_column", "word", "text column from the dataset")
#     flags.DEFINE_string("entity_column","entity_name","entity column from the dataset" )
#
#     # CoNLL formatted text files
#     flags.DEFINE_string("train_data_text_file","none","")
#     flags.DEFINE_string("val_data_text_file","none","")
#     flags.DEFINE_string("predict_data_text_file","none","")
#
#     # Domain specific input files
#     flags.DEFINE_string("train_csvs_path", "data/train/","")
#     flags.DEFINE_string("val_csvs_path","data/val/","")
#     flags.DEFINE_string("test_csvs_path", "data/test/","")
#     flags.DEFINE_string("db_reference_file", "data/desired_labels.csv","")
#
#     flags.DEFINE_string("id_field" ,"doc_id", "")
#
#     # Output directories
#     flags.DEFINE_string("data-dir","experiments/tf-data","")
#
#     flags.DEFINE_string("words_vocab_file","experiments/tf-data/word_vocab.tsv","")
#     flags.DEFINE_string("chars_vocab_file","experiments/tf-data/char_vocab.tsv","")
#     flags.DEFINE_string("entity_vocab_file","experiments/tf-data/entity_vocab.tsv","")
#
#
#     # Model output directories
#     flags.DEFINE_string("model_root_dir","experiments/tf-models/","")
#
#     flags.DEFINE_string("pre_trained_model_dir","","")
#
#     flags.DEFINE_string("prediction_root_dir","experiments/tf-out/","")
#     flags.DEFINE_string("prediction_input_csv_dir","experiments/tf-data/test-iob-annotated/","")
#     flags.DEFINE_string("trained_model_dir","none","")
#
#
#
#
#
#
#     # For Token
#
#     flags.DEFINE_integer("pad_word_id", 0,"")
#     flags.DEFINE_integer("unknown_word_id", 1,"")
#     flags.DEFINE_string("unknown_word", "<UNK>","")
#     flags.DEFINE_string("pad_word", "<PAD>","")
#     flags.DEFINE_string("space_appended_pad_token", " <PAD>","pad token with space")
#
#     flags.DEFINE_string("separator", " ","separator for token in files")
#     flags.DEFINE_string("quotechar", "^","quote char in files")
#
#     # for trainin,
#     flags.DEFINE_integer("batch_size", 12, "batch size")
#     flags.DEFINE_string("configure_model","yes","") #**** Set up if you need to retrain a model ********#
#     flags.DEFINE_integer("num_epochs", 1, "num epochs")
#
#     flags.DEFINE_string("model_name", "bilstm_crf_v0","")
#
#
#     flags.DEFINE_string("use_char_embedding",False,"")
#     flags.DEFINE_float("learning_rate", 0.002,"")
#     flags.DEFINE_integer("word_level_lstm_hidden_size", 300,"")
#     flags.DEFINE_integer("char_level_lstm_hidden_size",  300,"")
#     flags.DEFINE_integer("word_emd_size",300,"")
#     flags.DEFINE_integer("char_emd_size", 300,"")
#     flags.DEFINE_integer("num_lstm_layers", 2,"")
#     flags.DEFINE_float("out_keep_propability" , 0.75,"")
#     flags.DEFINE_boolean("use_crf", True,"")
#
#     # for files
#
#
#     flags.DEFINE_integer("test_batch_size" ,1,"")
#
#
#     ############################
#     #   environment setting    #
#     ############################
#     # flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist, fashion-mnist")
#     flags.DEFINE_boolean("is_training", True, "train or predict phase")
#     # flags.DEFINE_integer("num_threads", 8, "number of threads of enqueueing exampls")
#     # flags.DEFINE_string("logdir", "logdir", "logs directory")
#     # flags.DEFINE_integer("train_sum_freq", 100, "the frequency of saving train summary(step)")
#     # flags.DEFINE_integer("val_sum_freq", 500, "the frequency of saving valuation summary(step)")
#     # flags.DEFINE_integer("save_freq", 3, "the frequency of saving model(epoch)")
#     # flags.DEFINE_string("results", "results", "path for saving results")
#
#     ############################
#     #   distributed setting    #
#     ############################
#     # flags.DEFINE_integer("num_gpu", 2, "number of gpus for distributed training")
#     # flags.DEFINE_integer("batch_size_per_gpu", 128, "batch size on 1 gpu")
#     # flags.DEFINE_integer("thread_per_gpu", 4, "Number of preprocessing threads per tower.")
#
#
#     ############################
#     #   run setting    #
#     ############################
#
#
#     cfg = tf.app.flags.FLAGS
#     # tf.logging.set_verbosity(tf.logging.INFO)
#
#     return cfg