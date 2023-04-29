import csv
import torch
import numpy as np

def load_unsmile_data(train=True, return_column_name=True, return_torch=True):
    if train:
        file_name = "unsmile_dataset/unsmile_train_v1.0.tsv"
    else:
        file_name = "unsmile_dataset/unsmile_valid_v1.0.tsv"
    with open(file_name, encoding="utf-8") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        data = [line for line in tsv_file]
        column_name = data[0]
        data = data[1:]
    texts = [row[0] for row in data]
    labels = [int(label) for row in data for label in row[1:]]
    if return_torch:
        labels = torch.reshape(torch.tensor(labels, dtype=torch.float32), (-1, 11))
    else:
        labels = np.reshape(labels, (-1, 11)).astype("float32")
    if return_column_name:
        return texts, labels, column_name
    else:
        return texts, labels
