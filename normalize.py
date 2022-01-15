import os, sys
from conllu import parse
import pandas as pd
import string
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


out_folder = sys.argv[1]
os.makedirs(out_folder, exist_ok=True)

#Wiki
print("Normalize wiki")
wiki_path = "/home/tsargsyan/davit/arm/data/arm-wiki/tmp.txt"
texts = read_wiki(wiki_path)
n_texts = normalize_many(texts)
print("Normalized {} words".format(count_words(n_texts)))
write_texts(n_texts, os.path.join(out_folder, "arm-wiki.txt"))

#Conllu
print("Normalize Conllu")
train_conllu = "/home/tsargsyan/davit/arm/data/conllu/UD_Armenian-ArmTDP/hy_armtdp-ud-train.conllu"
dev_conllu = "/home/tsargsyan/davit/arm/data/conllu/UD_Armenian-ArmTDP/hy_armtdp-ud-dev.conllu"
test_conllu = "/home/tsargsyan/davit/arm/data/conllu/UD_Armenian-ArmTDP/hy_armtdp-ud-test.conllu"

for conllu_path, name in  zip([train_conllu, dev_conllu, test_conllu], ["train", "dev", "test"]):
    texts = read_conllu(conllu_path)
    n_texts = normalize_many(texts)
    print("Normalized {} words".format(count_words(n_texts)))
    write_texts(n_texts, os.path.join(out_folder, f"{name}-conllu.txt"))

#ARPA
print("Normalize ARPA")
train_arpa = "/home/tsargsyan/davit/arm/data/arpa/arpa-paraphrase-corpus/train.tsv"
test_arpa = "/home/tsargsyan/davit/arm/data/arpa/arpa-paraphrase-corpus/test.tsv"
for arpa_path, name in  zip([train_arpa, test_arpa], ["train", "test"]):
    texts = read_arpa(arpa_path)
    n_texts = normalize_many(texts)
    print("Normalized {} words".format(count_words(n_texts)))
    write_texts(n_texts, os.path.join(out_folder, f"{name}-arpa.txt"))