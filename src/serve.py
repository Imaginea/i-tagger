from taggers.conll_tagger import CoNLLTagger


def get_model_api(model_dir,abs_fpath):
    """Returns dataframe"""

    # 1. initialize model
    tagger = CoNLLTagger(model_dir=model_dir)
    a = tagger.predict_on_test_files(abs_fpath)
    return a

def get_model_api1(model_dir,sentence):
    """Returns dataframe"""

    # 1. initialize model
    tagger = CoNLLTagger(model_dir=model_dir)
    preds = tagger.predict_on_test_text(sentence)
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