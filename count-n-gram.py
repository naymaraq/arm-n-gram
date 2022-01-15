import os
import subprocess
import sys
import progressbar
import pandas as pd
from tqdm.auto import tqdm
import json
from tqdm import tqdm as tq
from collections import Counter
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from glob import glob

n = int(sys.argv[1])
save_dir = sys.argv[2]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def read_train_file(paths, lowercase: bool=False):
    lines_read = 0
    text_dataset = []
    for path in paths.split(","):
        with open(path, 'r') as f:
            reader = tqdm(iter(lambda: f.readline(), ''), desc="Read 0 lines", unit=' lines')
            for i, line in enumerate(reader):
                if path.endswith('.json'):
                    line = json.loads(line).get('text', None) or json.loads(line).get('transcript')

                line = line.replace("\n", "").strip()
                if lowercase:
                    line = line.lower()

                if line:
                    text_dataset.append(line)

                    lines_read += 1
                    if lines_read % 100000 == 0:
                        reader.set_description(f"Read {lines_read} lines")

    return text_dataset


def segment_text(text, n):
    text = text.split()
    return [' '.join(text[i:i+n]) for i in range(0,len(text),n)]

def all_data(dataset):
    all_data = []
    for data in dataset:
        texts  = read_train_file(data)
        for text in texts:
            if len(text.split()) >= 50:
                all_data.extend(segment_text(text,n=50))
            else:
                all_data.append(text)
    return  all_data


def tokens_count(texts):
    return sum(len(t.split(" ")) for t in texts)

dataset = glob("/home/tsargsyan/davit/arm/data/normalized/*")
texts = all_data(dataset)
print("Words count is:", tokens_count(texts))

indecies = []
word2index = {"last": 0}
for text in tq(texts):
    words = text.split(" ")
    sent_indecies = []
    for word in words:
        index = word2index.get(word, -1)
        if index == -1:
            index = word2index["last"]
            word2index[word] = index
            word2index["last"] = word2index["last"] + 1
        sent_indecies.append(index)
    indecies.append(sent_indecies)

max_len = 0
for arr in indecies:
    new_len = len(arr)
    if new_len >= max_len:
        max_len = new_len

desired_array = []
for array in tq(indecies):
    array += (max_len - len(array)) * [0]
    desired_array.append(array)

final_list = []
desired_array = np.array(desired_array)

new_dict = Counter({})
for i in tq(range(max_len + 1 - n)):
    cur_list = desired_array[:, i: i + n]
    cur_list = filter(lambda x: (0 not in x), cur_list)
    cur_list = list(map(tuple, cur_list))
    new_dict += Counter(cur_list)
new_dict = dict(new_dict)
dict_values = np.array(list(new_dict.values()))

arr = np.expand_dims(np.arange(max(dict_values)), -1)
batch_size = 50000
b = []
for i in tqdm(range(int(len(dict_values) / batch_size) + 1)):
    b.append(np.sum(dict_values[i * batch_size: (i + 1) * batch_size] >= arr, axis=-1))
    counts = np.sum(np.array(b), axis=0)
    plot({"data": [go.Scatter(x=np.arange(len(counts)), y=counts)],
          "layout": go.Layout(title="Distribution of %s grams"%n)},
          image='jpeg', filename=os.path.join(save_dir, "%s_grams.html"%n))
