import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#==================================================================================================
# Predictions Post processing
#==================================================================================================

def is_prediction_matching(x, y):
    result = False
    if len(x)==1 or len(y)==1:
        return False
    elif "-" not in x or "-" not in y:
        return False
    else:
        return x.split("-")[1] == y.split("-")[1]


def get_naive_metrics(predicted_csvs_path, ner_tag_vocab_file, entity_col_name, prediction_col_name, out_dir):

    out_dir = out_dir + "/metrics/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(ner_tag_vocab_file, 'r') as file:
        ner_vocab = [tag.strip() for tag in file.readlines()]
        ner_vocab = set(ner_vocab)
        ner_vocab = list(ner_vocab)

    dfs = []
    for csv_file in tqdm(os.listdir(predicted_csvs_path)):
        csv_file = os.path.join(predicted_csvs_path, csv_file)
        if csv_file.endswith(".csv"):
            df = pd.read_csv(csv_file)
            df = df[[entity_col_name, prediction_col_name]]
            df["similarity"] = df[entity_col_name] == df[prediction_col_name]
            df["similarity"] = df["similarity"].astype(int)
            dfs.append(df)
    final_df = pd.concat(dfs)

    result_rows = []
    for current_ner_tag in ner_vocab:
        filtered_df = final_df[final_df[entity_col_name] == current_ner_tag]
        if current_ner_tag == "O":
            row = [current_ner_tag, \
                   filtered_df.count()[0], \
                   filtered_df["similarity"].sum(), \
                   (filtered_df["similarity"].sum() / filtered_df.count()[0]) * 100, \
                   filtered_df.count()[0] - filtered_df["similarity"].sum(), \
                   filtered_df[filtered_df["predictions"] != "O"].count()[0]]
        else:
            row = [current_ner_tag, \
                   filtered_df.count()[0], \
                   filtered_df["similarity"].sum(), \
                   (filtered_df["similarity"].sum() / filtered_df.count()[0]) * 100, \
                   filtered_df.count()[0] - filtered_df["similarity"].sum(), \
                   filtered_df[filtered_df["predictions"] == "O"].count()[0]]
        result_rows.append(row)
    metric_df = pd.DataFrame(result_rows, columns=["Label_name",
                                                   "actual_count_of_labels",
                                                   "Predicted_count_labels",
                                                   "Accuracy",
                                                   "Missed_prediction_count",
                                                   "Wrong_prediction_count"])

    metric_df.to_csv(out_dir + "/metric_report.csv", index=False)

    return metric_df


