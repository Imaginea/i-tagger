import pandas as pd
import os
from tqdm import tqdm
TRAIN_FILE = "conll_experiments/data/train.txt"
TEST_FILE = "conll_experiments/data/test.txt"
VAL_FILE = "conll_experiments/data/val.txt"

TRAIN_FILE_OUT = "conll_csv_experiments/data/train/"
TEST_FILE_OUT = "conll_csv_experiments/data/test/"
VAL_FILE_OUT = "conll_csv_experiments/data/val/"

def conll_to_csv(file_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(file_path, sep=" ", skip_blank_lines=False, header=None).fillna("<UNK>")
    df = df[~df[0].str.contains("DOCSTART")]
    current_file = []
    for i in tqdm(range(len(df))):
        row = df.values[i]
        if row[0] != "<UNK>":
            current_file.append(row)
        else:
            if len(current_file) > 1:
                current_file = pd.DataFrame(current_file)
                current_file.to_csv(out_dir + "/{}.csv".format(i), index=False)
                current_file = []


conll_to_csv(TRAIN_FILE, TRAIN_FILE_OUT)
conll_to_csv(TEST_FILE, TEST_FILE_OUT)
conll_to_csv(VAL_FILE, VAL_FILE_OUT)


