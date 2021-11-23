import pandas as pd
import matplotlib as plt

train_path = "./data/train.txt"
dev_path = "./data/dev.txt"

def read_text(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) == 3:
                yield {'query1' : data[0], 'query2' : data[1], "label" : data[2]}
            else:
                yield {'query1' : data[0], 'query2' : data[1]}

df_train = read_text(train_path)
df_dev = read_text(dev_path)