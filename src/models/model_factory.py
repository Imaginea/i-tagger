import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class TFEstimatorFactory():

    models = {
        "bilstm_crf_v0": "BiLSTMCRFV0"
    }

    def __init__(self):
        ""

    @staticmethod
    def _get_model(name):
        '''
        '''
        try:
            cfg = getattr(import_module("models." + name), TFEstimatorFactory.models[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return cfg


    @staticmethod
    def get(model_name="bilstm_crf_v0"):
        model = TFEstimatorFactory._get_model(model_name)
        return model


