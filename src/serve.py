from config.global_constants import UNKNOWN_WORD
from data_iterators.data_iterators_factory import DataIteratorsFactory
from helpers.print_helper import *
from commands.tagger import load_estimator

import pandas as pd

def get_model_api(model_dir, abs_fpath):
    """Returns dataframe"""

    # 1. initialize model
    decoded_path = model_dir.split("/")
    model_name = decoded_path[-2]
    data_iterator_name = decoded_path[-3]
    experiment_name = decoded_path[-4]

    estimator = load_estimator(experiment_name=experiment_name,
                               data_iterator=data_iterator_name,
                               model_name=model_name,
                               model_dir=model_dir)


    # Use the factory loader to load the appropriate data-iterator,
    # which depends/reads the config from experiment_folder/config/*.pickle
    # stored previously by preprocessor
    data_iterator = DataIteratorsFactory.get(data_iterator_name)
    # Initialize the data iterator with experiment folder path and batch size
    # all other needed config/info are read from the *.pickle file
    df = None
    if abs_fpath.endswith(".csv"):
        df = pd.read_csv(abs_fpath).fillna(UNKNOWN_WORD)
    elif abs_fpath.endswith(".json"):
        df = pd.read_json(abs_fpath).filla(UNKNOWN_WORD)

    a = data_iterator.predict_on_test_file(estimator=estimator, df=df)
    return a

def get_model_api1(model_dir, sentence):
    """Returns dataframe"""

    # 1. initialize model
    print_info(model_dir)

    decoded_path = model_dir.split("/")
    model_name = decoded_path[-2]
    data_iterator_name = decoded_path[-3]
    experiment_name = decoded_path[-4]

    print_info(model_name)
    print_info(data_iterator_name)
    print_info(experiment_name)
    estimator = load_estimator(experiment_name=experiment_name,
                               data_iterator=data_iterator_name,
                               model_name=model_name,
                               model_dir=model_dir)


    # Use the factory loader to load the appropriate data-iterator,
    # which depends/reads the config from experiment_folder/config/*.pickle
    # stored previously by preprocessor
    data_iterator = DataIteratorsFactory.get(data_iterator_name)
    # Initialize the data iterator with experiment folder path and batch size
    # all other needed config/info are read from the *.pickle file
    data_iterator = data_iterator(experiment_name, -1)

    preds = data_iterator.predict_on_text(estimator, sentence)
    # 2. process input
    punc = [",", "?", ".", ":", ";", "!", "(", ")", "[", "]"]
    s = "".join(c for c in sentence if c not in punc)
    words_raw = s.strip().split(" ")

    # 4. process the output
    print(preds)
    print(words_raw)
    output_data = align_data({"input": words_raw, "output": preds})
    return output_data

def align_data(data):
    """Given dict with lists, creates aligned strings
    Args:
        data: (dict) data["x"] = ["I", "love", "India"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love India"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned