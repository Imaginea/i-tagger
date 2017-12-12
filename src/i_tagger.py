import sys
sys.path.append("../")

import tensorflow as tf

def get_tf_flags():

    flags = tf.app.flags

    flags.DEFINE_string("action","none","preprocess/train/retrain")

    flags.DEFINE_string("data_dir","experiments/tf_data/","")


    cfg = tf.app.flags.FLAGS
    return cfg

class ITagger():
    def __init__(self, tf_flags):
        self.preprocessor = None

    def add_preprocessor(self, preprocessor):
        '''
        
        :return: 
        '''
        self.preprocessor = preprocessor()

    def add_data_iterator(self, data_iterator):
        self.data_iterator = data_iterator()

    def add_estimator(self, estimator):
        self.estimator = estimator()