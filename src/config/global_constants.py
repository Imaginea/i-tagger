from config.config_helper import ConfigManager

global_constants = ConfigManager("src/config/global_constants.ini")

PAD_WORD = global_constants.get_item("VOCAB", "unknown_word")
UNKNOWN_WORD = global_constants.get_item("VOCAB", "padding_word")
PAD_WORD_ID = global_constants.get_item("VOCAB", "unknown_word_id")
UNKNOWN_WORD_ID = global_constants.get_item("VOCAB", "padding_word_id")

PAD_CHAR = global_constants.get_item("VOCAB", "unknown_char")
UNKNOWN_CHAR = global_constants.get_item("VOCAB", "padding_char")
PAD_CHAR_ID = global_constants.get_item("VOCAB", "unknown_char_id")
UNKNOWN_CHAR_ID = global_constants.get_item("VOCAB", "padding_char_id")

SEPRATOR = " "
QUOTECHAR = "^"
