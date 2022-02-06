from conllu import parse
import pandas as pd
import unicodedata
import re

alphabet = {"ց", "ր","փ","ւ","օ","ք","և","ֆ","ՈՒ",
            "Ու","ու","Ա","Գ","Բ","Ե","Դ","Է","Զ",
            "Թ","Ը","Ի","Ժ","Խ","Լ","Կ","Ծ","Ձ","Հ","Ճ","Ղ","Յ",
            "Մ","Շ","Ն","Չ","Ո","Ջ","Պ","Ս","Ռ","Տ","Վ","Ց",
            "Ր","Փ","Ւ","Օ","Ք","Ֆ","ա","գ","բ","ե","դ","է",
            "զ","թ","ը","ի","ժ","խ","լ","կ","ծ","ձ","հ","ճ","ղ",
            "յ","մ","շ","ն","չ","ո","ջ","պ","ս","ռ","տ","վ",
            "0", "1","2","3", "4", "5", "6", "7", "8", "9", " "
            }

def read_wiki(text_path):
    texts = []
    with open(text_path) as f:
        for line in f.readlines():
            texts.append(line)
    return texts

def read_conllu(conllu_path):
    with open(conllu_path) as f:
        data = ''.join(f.readlines())
        f.close()
    sentences = parse(data)
    texts = []
    for sentense in sentences:
        text = sentense.metadata['text']
        texts.append(text)
    return texts

def read_arpa(arpa_path):

    df = pd.read_csv(arpa_path, sep="\t")[["Sentence1","Sentence2"]]
    texts = []
    for i, row in df.iterrows():
        texts.append(row[0])
        texts.append(row[1])
    return texts

def normalize_one(text):
    text = text.lower()
    diff = set(text) - alphabet
    replace_table = str.maketrans({ch: None for ch in diff})
    text = text.translate(replace_table)

    #Unicode Normalization
    text = re.sub("[ ]{2,}", " ", text)
    text = (unicodedata.normalize("NFKD", text)
             .encode("utf-8", "ignore")
             .decode("utf-8", "ignore")
             )
    return text

def count_words(texts):
    return sum((len(t.split()) for t in texts))

def normalize_many(texts):
    normalized_texts = []
    for text in texts:
        n = normalize_one(text)
        normalized_texts.append(n)
    return normalized_texts

def write_texts(texts, out_path):
    with open(out_path,"w") as f:
        for text in texts:
            f.write(text)
            f.write("\n")