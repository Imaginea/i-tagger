import sys
sys.path.append("../")

from config.config_factory import TFConfigFactory
class ITagger():
    def __init__(self, config_name):
        self.config = TFConfigFactory.get(config_name=config_name)

        self.preprocessor = None

    def add_preprocessor(self, preprocessor):
        '''
        
        :return: 
        '''
        self.preprocessor = preprocessor(self.config)

    def add_data_iterator(self):
        raise NotImplementedError

    def add_estimator(self):
        raise NotImplementedError