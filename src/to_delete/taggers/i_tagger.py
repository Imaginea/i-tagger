import sys
sys.path.append("../")

import tensorflow as tf


class ITagger():
    def __init__(self):
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