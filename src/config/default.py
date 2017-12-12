# from config.config_helper import *
#
# # may be move to class
# def create_default_config(config_manager: ConfigManager):
#     schema_config_manager = SchemaConfigHelper(config_manager)
#     token_config_manager = TokenConfigHelper(config_manager)
#     parameter_config_manager = ParameterConfigHelper(config_manager)
#
#     # region Token
#     # Section "|Token|"
#     token_config_manager.set_pad_word_id(0)
#     token_config_manager.set_unknown_word_id(1)
#     token_config_manager.set_unknown_word("<UNK>")
#     token_config_manager.set_pad_word("<PAD>")
#     # endregion
#
#     # region Schema
#     # text_column = "word"
#     schema_config_manager.set_text_column("word")
#
#     # entity_column = "entity_name"
#     schema_config_manager.set_entity_column("entity_name")
#
#     # doc_id
#     schema_config_manager.set_id_field("doc_id")
#     # endregion
#
#     # region Data
#     # Section "|Data|"
#     # config_manager.add_section("Data")
#
#     # UNKNOWN_WORD = "<UNK>"
#     # config_manager.set_item("Data", "UNKNOWN_WORD", "<UNK>")
#     #
#     # # PAD_WORD = "<PAD>"
#     # config_manager.set_item("Data", "PAD_WORD", "<PAD>")
#     # endregion
#
#     # region Files
#     # Section "|Files|"
#     config_manager.add_section("Files")
#
#     # train_csvs_path = "../data/build/train/"
#     config_manager.set_item("Files", "train_csvs_path", "../data/build/train/")
#
#     # val_csvs_path = "../data/build/val/"
#     config_manager.set_item("Files", "val_csvs_path", "../data/build/val/")
#
#     # predict_csvs_path = "../data/build/test/"
#     config_manager.set_item("Files", "predict_csvs_path", "../data/build/test/")
#
#     # db_reference_file = "../data/build/desired_labels.csv"
#     config_manager.set_item("Files", "db_reference_file", "../data/build/desired_labels.csv")
#
#     # CoNLL
#     # train_data_text_file = ""
#     config_manager.set_item("Files", "train_data_text_file", "none")
#
#     # val_data_text_file = ""
#     config_manager.set_item("Files", "val_data_text_file", "none")
#
#     # predict_data_text_file = ""
#     config_manager.set_item("Files", "predict_data_text_file", "none")
#
#     # data-dir ="experiments/tf-data"
#     config_manager.set_item("Files", "data-dir", "experiments/tf-data")
#
#     # self.WORDS_VOCAB_FILE = opt.data_dir + "/" + self.TEXT_COL + "_" + "tokens_vocab.tsv"
#     config_manager.set_item("Files", "words_vocab_file", "${data-dir}/${Schema:text_column}_tokens_vocab.tsv")
#
#     # self.CHARS_VOCAB_FILE = opt.data_dir + "/" + self.TEXT_COL + "_" + "chars_vocab.tsv"
#     config_manager.set_item("Files", "chars_vocab_file", "${data-dir}/${Schema:text_column}_chars_vocab.tsv")
#
#     # self.ENTITY_VOCAB_FILE = opt.data_dir + "/" + self.ENTITY_COL + "_vocab.tsv"
#     config_manager.set_item("Files", "entity_vocab_file", "${data-dir}/${Schema:entity_column}_vocab.tsv")
#
#     # endregion
#
#     # region Run
#     # Section "|Run|"
#     config_manager.add_section("Run")
#
#     # over_write= "yes"
#     config_manager.set_item("Run", "over_write", "yes")
#
#     # use_iob_format = "yes"
#     config_manager.set_item("Run", "use_iob_format", "yes")
#     # endregion
#
#     # region Parameter Train
#
#     parameter_config_manager.set_batch_size(4)
#
#     parameter_config_manager.set_is_model_configure("no")
#
#     parameter_config_manager.set_model_name("bilstm_crf_v0")
#
#     parameter_config_manager.set_num_epochs(3)
#
#     parameter_config_manager.set_use_char_embedding("no")
#
#     config_manager.set_item("Files", "model_root_dir", "experiments/tf-models/")
#
#     config_manager.set_item("Files", "pre_trained_model_dir", "none")
#
#     parameter_config_manager.set_learning_rate(0.002)
#     parameter_config_manager.set_word_level_lstm_hidden_size(300)
#     parameter_config_manager.set_char_level_lstm_hidden_size(300)
#     parameter_config_manager.set_word_emd_size(300)
#     parameter_config_manager.set_char_emd_size(300)
#     parameter_config_manager.set_num_lstm_layers(2)
#     parameter_config_manager.set_out_keep_propability(0.75)
#     parameter_config_manager.set_use_crf(True)
#
#     # endregion
#
#     # region Prediction
#     # model_name = bilstm_crf_v0 already set
#     config_manager.set_item("Files", "prediction_root_dir", "experiments/tf-out/")
#     config_manager.set_item("Files", "prediction_input_csv_dir", "experiments/tf-data/test-iob-annotated/")
#     config_manager.set_item("Files", "trained_model_dir", "${Files:model_root_dir}experiments/tf-models/"+parameter_config_manager.get_model_name()+"/default/")
#     # endregion
#     #
#     #TODO convert to int
#     config_manager.add_section("Predict")
#     config_manager.set_item("Predict", "train_batch_size", str(1)) #TODO fix this
#     config_manager.save_config()
