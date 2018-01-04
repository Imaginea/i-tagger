import sys
sys.path.append("../")
from config.config_helper import ConfigManager

class IPreprocessorInterface():
    '''
    Preprocesing stage of the trianing, 
    where the data is transformed from one form to another.
    
    It is expected the data resides in its own experiment folder, along
    with config for supported preprocessor.
    '''
    def __init__(self, experiment_root_directory):
        '''
        Some rules are enforced here, such that it follows following directory structure
        experiment_name/
            preprocessed_data/
                train/
                val/
                test/
        :param experiment_root_directory: 
        '''

        self.EXPERIMENT_ROOT_DIR = experiment_root_directory

        self._load_ini()

        # If this rule changes, make relevant changes in `IDataIterator` also!
        self.PREPROCESSED_DATA_DIR = self.EXPERIMENT_ROOT_DIR + "/" + self.config.get_item("OutputDirectories","preprocessed_data_dir")

        self.TRAIN_OUT_PATH = self.PREPROCESSED_DATA_DIR + "/train/"
        self.VAL_OUT_PATH = self.PREPROCESSED_DATA_DIR + "/val/"
        self.TEST_OUT_PATH = self.PREPROCESSED_DATA_DIR + "/test/"


    def _load_ini(self):
        '''
        Assuming each dataset will have its own configuration, a `experiment_folder/config/*.ini`
        is used to store and read data specific configuration
        :return: 
        '''
        self.config = ConfigManager(self. EXPERIMENT_ROOT_DIR + "/config/config.ini")

    def _create_target_directories(self):
        '''
        We wish not to preprocess the data every time, so before storing the preprocessed
        data appropriate data folders are created and **over write** should be handled
        :return: 
        '''
        raise NotImplementedError

    def _prepare_data(self):
        raise NotImplementedError


    def preprocess(self):
        self._create_target_directories()
        self._prepare_data()



