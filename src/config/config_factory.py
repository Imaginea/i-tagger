import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class TFConfigFactory():
    def __init__(self):
        self.config = None

    @staticmethod
    def _get_config_module(name):
        '''
        Import the module of 'name' dynamically
        :param name: Config file name
        :return: 
        '''
        try:
            cfg = getattr(import_module("config." + name), "get_global_config")
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return cfg

    @staticmethod
    def reset():
        tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()

    @staticmethod
    def get(config_name="default"):
        '''
        Gets the TensorFlow runtime config, defined in src/config/ path
        :param config_name: Name of the file to be used for global config
        :return: 
        '''
        get_global_config = TFConfigFactory._get_config_module(config_name)
        return get_global_config()


