import os
import configparser
from configparser import ExtendedInterpolation

class ConfigManager(object):
    def __init__(self, config_path: str):
        # set the path to the config file
        self.config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
        self.config_path = config_path

        if os.path.exists(config_path):
            self.config.read(self.config_path)
        else:
            self.save_config()

    def set_item(self, section: str, option: str, value: str):
        self.config.set(section=section,
                        option=option,
                        value=value)

    def get_item(self, section, option)-> str:
        return self.config.get(section=section,
                               option=option)
    def add_section(self, section):
        self.config.add_section(section)

    def get_item_as_int(self,section, option):
        return self.config.getint(section=section,
                               option=option)

    def get_item_as_boolean(self,section, option):
        return self.config.getboolean(section=section,
                               option=option)

    def save_config(self):
        # Writing our configuration file to 'config'
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

    # def create_default_config(self):
    #     UNKNOWN_WORD = "<UNK>"
    #     self.config.set_item("Data", "UNKNOWN_WORD", "<UNK>")
    #
    #     PAD_WORD = "<PAD>"
    #     self.config.set_item("Data", "PAD_WORD", "<PAD>")


class SchemaConfigHelper(object):
    def __init__(self, config_helper: ConfigManager):
        self.config = config_helper
        #TODO check if the Schema Section is present
        #if absent create
        if "Schema" not in self.config.config.sections():
            self.config.add_section("Schema")

    def set_schema_fields(self, feature_fields:[]):
        self.config.set_item("Schema", "Fields", ",".join(feature_fields))

    def get_schema_fields(self):
        value = self.config.get_item("Schema", "Fields")
        return [chunk.strip() for chunk in value.split(",")]

    def get_id_field(self):
        return self.config.get_item("Schema", "id_field")

    def set_id_field(self,id_field):
        self.config.set_item("Schema", "id_field", id_field)
    #
    # def get_entity_iob_col(self):
    #     return self.config.get_item("Schema", "entity_iob_col")
    #
    # def get_class_field(self):
    #     return self.config.get_item("Schema", "iob")
    #
    # def set_class_field(self):
    #     pass

    def set_text_column(self,text_column)-> str:
        return self.config.set_item("Schema", "text_column", text_column)

    def set_entity_column(self,entity_column)-> str:
        return self.config.set_item("Schema", "entity_column", entity_column)

    def get_text_column(self)-> str:
        return self.config.get_item("Schema", "text_column")

    def get_entity_column(self)-> str:
        return self.config.get_item("Schema", "entity_column")

    def get_entity_iob_column(self) -> str:
        return self.config.get_item("Schema", "entity_column")+ "_iob"

#TokenConfigHelper

class TokenConfigHelper(object):
    def __init__(self, config_helper: ConfigManager):
        self.config = config_helper
        if "Token" not in self.config.config.sections():
            self.config.add_section("Token")
    # TODO check if the Schema Section is present
    # if absent create
    def set_pad_word_id(self,pad_word_id:int):
        self.config.set_item("Token", "PAD_WORD_ID", str(pad_word_id))

    def get_pad_word_id(self):
        return self.config.get_item_as_int("Token", "PAD_WORD_ID")

    def set_unknown_word_id(self, unknown_word_id:int):
        self.config.set_item("Token", "UNKNOWN_WORD_ID", str(unknown_word_id))

    def get_unknown_word_id(self):
        return self.config.get_item_as_int("Token", "UNKNOWN_WORD_ID")

    #TODO Not pushing these in config ..may be later
    #=========================================
    def get_separator(self):
        return " "

    def get_quote_character(self):
        return "^"
    #========================================
    # UNKNOWN_WORD = "<UNK>"
    def set_unknown_word(self, unknown_word: int):
        self.config.set_item("Token", "UNKNOWN_WORD", unknown_word)

    def get_unknown_word(self):
        return  self.config.get_item("Token", "UNKNOWN_WORD")

    #  PAD_WORD = "<PAD>"
    def set_pad_word(self, pad_word: int):
        self.config.set_item("Token", "PAD_WORD", pad_word)

    def get_pad_word(self):
        return  self.config.get_item("Token", "PAD_WORD")


class ParameterConfigHelper(object):
    def __init__(self, config_helper: ConfigManager):
        self.config = config_helper
        #TODO check if the Schema Section is present
        #if absent create
        if "Parameter" not in self.config.config.sections():
            self.config.add_section("Parameter")

    def set_model_name(self,model_name):
        self.config.set_item("Parameter", "model_name", model_name)

    def get_model_name(self):
        return self.config.get_item("Parameter", "model_name")

    def set_use_char_embedding(self,use_char_embedding):
        self.config.set_item("Parameter", "use_char_embedding", use_char_embedding)

    def get_use_char_embedding(self):
        return self.config.get_item_as_boolean("Parameter", "use_char_embedding")

    # TODO this should be int
    def set_num_epochs(self,num_epochs):
        self.config.set_item("Parameter", "num_epochs", str(num_epochs))

    def get_num_epochs(self):
        return self.config.get_item_as_int("Parameter", "num_epochs")

    def set_batch_size(self,batch_size):
        self.config.set_item("Parameter", "batch_size", str(batch_size))

    def get_batch_size(self):
        return self.config.get_item_as_int("Parameter", "batch_size")

    def set_is_model_configure(self,configure_model):
        self.config.set_item("Parameter", "configure_model", configure_model)

    def get_is_model_configure(self):
        return self.config.get_item_as_boolean("Parameter", "configure_model")

    def set_learning_rate(self,lr):
        self.config.set_item("Parameter", "learning_rate", str(lr))

    def set_word_level_lstm_hidden_size(self,word_level_lstm_hidden_size):
        self.config.set_item("Parameter", "word_level_lstm_hidden_size", str(word_level_lstm_hidden_size))

    def set_char_level_lstm_hidden_size(self, char_level_lstm_hidden_size):
        self.config.set_item("Parameter", "char_level_lstm_hidden_size", str(char_level_lstm_hidden_size))

    def set_word_emd_size(self,word_emd_size):
        self.config.set_item("Parameter", "word_emd_size", str(word_emd_size))

    def set_char_emd_size(self,char_emd_size):
        self.config.set_item("Parameter", "char_emd_size", str(char_emd_size))

    def set_num_lstm_layers(self,num_lstm_layers):
        self.config.set_item("Parameter", "num_lstm_layers", str(num_lstm_layers))

    def set_out_keep_propability(self, out_keep_propability):
        self.config.set_item("Parameter", "out_keep_propability", str(out_keep_propability))

    def set_use_crf(self,use_crf):
        self.config.set_item("Parameter", "use_crf", str(use_crf))
#====================================================================
    def get_learning_rate(self):
        self.config.get_item("Parameter", "learning_rate")

    def get_word_level_lstm_hidden_size(self):
        self.config.set_item("Parameter", "word_level_lstm_hidden_size")

    def get_char_level_lstm_hidden_size(self):
        self.config.set_item("Parameter", "char_level_lstm_hidden_size")

    def get_word_emd_size(self):
        self.config.set_item("Parameter", "word_emd_size")

    def get_char_emd_size(self):
        self.config.set_item("Parameter", "char_emd_size")

    def get_num_lstm_layers(self):
        self.config.set_item("Parameter", "num_lstm_layers")

    def get_out_keep_propability(self):
        self.config.set_item("Parameter", "out_keep_propability")

    def get_use_crf(self):
        self.config.set_item("Parameter", "use_crf")

#FileConfigHelper


# class FileConfigHelper(object):
#     def __init__(self, config_helper: ConfigHelper):
#         self.config = config_helper
#         if "Files" not in self.config.config.sections():
#             self.config.add_section("Files")
#     #TODO check if the Schema Section is present
#     #if absent create
#
#     def set_unknown_word(self,train_csvs_path):
#         self.config.set_item("Files", "train_csvs_path", train_csvs_path)
#
#     def set_unknown_word(self,val_csvs_path):
#     # val_csvs_path = "../data/build/val/"
#         self.config.set_item("Files", "val_csvs_path", val_csvs_path)
#
#     def set_unknown_word(self,predict_csvs_path):
#     # predict_csvs_path = "../data/build/test/"
#         self.config.set_item("Files", "predict_csvs_path", predict_csvs_path)
#
#     def set_unknown_word(self,db_reference_file):
#     # db_reference_file = "../data/build/desired_labels.csv"
#         self.config.set_item("Files", "db_reference_file", db_reference_file)
#
#     def set_unknown_word(self,train_data_text_file):
#     # CoNLL
#     # train_data_text_file = ""
#         self.config.set_item("Files", "train_data_text_file",train_data_text_file)
#
#     def set_unknown_word(self,val_data_text_file):
#     # val_data_text_file = ""
#         self.config.set_item("Files", "val_data_text_file", val_data_text_file)
#
#     def set_unknown_word(self,predict_data_text_file):
#     # predict_data_text_file = ""
#         self.config.set_item("Files", "predict_data_text_file", predict_data_text_file)
#
#     def set_unknown_word(self,data_dir):
#     # data-dir ="experiments/tf-data"
#         self.config.set_item("Files", "data-dir", data_dir)
