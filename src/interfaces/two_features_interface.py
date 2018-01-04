import sys
sys.path.append("../")

class IFeature():
    def __init__(self):
        self.NUM_FEATURES = None
        self.FEATURE_NAME = None


    def __eq__(self, other):
        """Override the default Equals behavior"""
        return self.FEATURE_NAME == other.FEATURE_NAME and self.NUM_FEATURES==other.NUM_FEATURES

#TODO Change name to ITextFeature
class ITextFeature(IFeature):
    def __init__(self):
        super(ITextFeature, self).__init__()
        self.FEATURE_NAME = "text+char_ids"
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"

        #TODO replace out put in data iterators and models
        self.OUT_TAGS = "classes"
        self.OUT_CONFIDENCE = "confidence"


class IPostionalFeature(IFeature):
    def __init__(self):
        self.FEATURE_NAME = "text+char_ids+positional_info"
        self.NUM_FEATURES = 3
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"
        self.FEATURE_3_NAME = "position"