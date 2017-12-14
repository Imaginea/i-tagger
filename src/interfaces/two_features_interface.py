import sys
sys.path.append("../")


class ITwoFeature():
    def __init__(self):
        self.FEATURE_TYPE = "text+char_ids"
        self.NUM_FEATURES = 2
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"