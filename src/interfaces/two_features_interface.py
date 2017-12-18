import sys
sys.path.append("../")

class TwoFeatures():
    def __init__(self):
        self.NUM_FEATURES = 2
        self.FEATURE_NAME = None
        self.FEATURE_1_NAME = None
        self.FEATURE_2_NAME = None

    # def equal:

#TODO Change name to ITextFeature
class ITextFeature(TwoFeatures):
    def __init__(self):
        super(ITextFeature, self).__init__()
        self.FEATURE_NAME = "text+char_ids"
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"

class IPostionalFeature():
    def __init__(self):
        self.FEATURE_TYPE = "text+char_ids+positional_info"
        self.NUM_FEATURES = 3
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"
        self.FEATURE_3_NAME = "x_cord,y_cord,page_no"