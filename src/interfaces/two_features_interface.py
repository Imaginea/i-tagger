import sys
sys.path.append("../")

class TwoFeatures():
    def __init__(self):
        self.NUM_FEATURES = 2
        self.FEATURE_NAME = None
        self.FEATURE_1_NAME = None
        self.FEATURE_2_NAME = None

    def __eq__(self, other):
        """Override the default Equals behavior"""
        return self.FEATURE_NAME == other.FEATURE_NAME and self.NUM_FEATURES==other.NUM_FEATURES

#TODO Change name to ITextFeature
class ITextFeature(TwoFeatures):
    def __init__(self):
        super(ITextFeature, self).__init__()
        self.FEATURE_NAME = "text+char_ids"
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"

        self.OUT_TAGS = "classes"
        self.OUT_CONFIDENCE = "confidence"


class IPostionalFeature():
    def __init__(self):
        self.FEATURE_NAME = "text+char_ids+positional_info"
        self.NUM_FEATURES = 3
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"
        self.FEATURE_3_NAME = "position"