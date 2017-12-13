import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class TFEstimatorFactory():

    models = {
        "bilstm_crf_v0": "BiLSTMCRFV0"
    }

    model_configurations = {
        "bilstm_crf_v0": "BiLSTMCRFConfigV0"
    }
    def __init__(self):
        ""

    @staticmethod
    def _get_model(name):
        '''
        '''
        try:
            model = getattr(import_module("models." + name), TFEstimatorFactory.models[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return model

    @staticmethod
    def _get_model_config(name):
        '''
        '''
        try:
            cfg = getattr(import_module("models." + name), TFEstimatorFactory.model_configurations[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return cfg


    @staticmethod
    def get(model_name="bilstm_crf_v0"):
        cfg = TFEstimatorFactory._get_model_config(model_name)
        model = TFEstimatorFactory._get_model(model_name)
        return cfg, model


