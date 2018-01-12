from config.config_helper import ConfigManager

global_constants = ConfigManager("src/config/global_constants.ini")

PAD_WORD = global_constants.get_item("VOCAB", "padding_word")
UNKNOWN_WORD = global_constants.get_item("VOCAB", "unknown_word")

PAD_WORD_ID = global_constants.get_item("VOCAB", "padding_word_id")
UNKNOWN_WORD_ID = global_constants.get_item("VOCAB", "unknown_word_id")

PAD_CHAR = global_constants.get_item("VOCAB", "padding_char")
UNKNOWN_CHAR = global_constants.get_item("VOCAB", "unknown_char")

PAD_CHAR_ID = global_constants.get_item("VOCAB", "padding_char_id")
UNKNOWN_CHAR_ID = global_constants.get_item("VOCAB", "unknown_char_id")

SEPERATOR = "~" # potential error point depending on the dataset
QUOTECHAR = "^"

EMPTY_LINE_FILLER = "<LINE_END>"

MAX_WORD_LENGTH = 20

PAD_POS = global_constants.get_item("VOCAB", "padding_pos")
PAD_POS_ID = global_constants.get_item("VOCAB", "padding_pos_id")

UNKNOWN_POS = global_constants.get_item("VOCAB", "unknown_pos")
UNKNOWN_POS_ID = global_constants.get_item("VOCAB", "unknown_pos_id")




