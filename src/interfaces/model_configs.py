import sys
sys.path.append("../")
import pickle
from helpers.print_helper import *

class IModelConfig():
    def __init__(self):
        ''

    @staticmethod
    def save(config, model_dir):
        print_info("Storing the ModelConfig for further use... \n{}\n ".format(config))

        with open(model_dir + "/model_config.pickle", "wb") as file:
            pickle.dump(config, file=file)

    @staticmethod
    def load(model_dir):
        config = None
        try:
            with open(model_dir + "/model_config.pickle", "rb") as file:
                config = pickle.load(file)
            print_info("Restoring the ModelConfig for further use... \n{}\n ".format(config))
        except:
            print_warn("No models at {}".format(model_dir))

        return config