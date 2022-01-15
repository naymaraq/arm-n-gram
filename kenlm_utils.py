import json
import os

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


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


def tokenize_str(texts, tokenizer, offset):
    tokenized_text = []
    for text in texts:
        tok_text = tokenizer.encode_as_ids(text)
        tok_text = [chr(token + offset) for token in tok_text]
        tokenized_text.append(tok_text)
    return tokenized_text


def tokenize_text(data, tokenizer, path, chunk_size=8192, buffer_size=32, token_offset=100):
    dataset_len = len(data)
    print(
        f"Chunking {dataset_len} rows into {dataset_len / float(chunk_size):0.4f} tasks (each chunk contains {chunk_size} elements)"
    )

    current_step = 0
    if os.path.exists(path):
        print(f"Deleting previous file : {path}")
        os.remove(path)

    with Parallel(n_jobs=-2, verbose=10) as parallel:
        while True:
            start = current_step * chunk_size
            end = min((current_step + buffer_size) * chunk_size, dataset_len)

            tokenized_data = parallel(
                delayed(tokenize_str)(data[start : start + chunk_size], tokenizer, token_offset)
                for start in range(start, end, chunk_size)
            )

            # Write dataset
            write_dataset(tokenized_data, path)
            current_step += len(tokenized_data)
            print(f"Finished writing {len(tokenized_data)} chunks to {path}. Current chunk index = {current_step}")
            del tokenized_data
            if end >= dataset_len:
                break


def write_dataset(chunks, path):
    basedir = os.path.dirname(path)

    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

    with open(path, 'a+', encoding='utf-8') as f:
        for chunk_idx in tqdm(range(len(chunks)), desc='Chunk ', total=len(chunks), unit=' chunks'):
            for text in chunks[chunk_idx]:
                line = ' '.join(text)
                f.write(f"{line}\n")