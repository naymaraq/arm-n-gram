import matplotlib.pyplot as plt
import os
import kenlm
import numpy as np

def read(text_path):
    texts = []
    with open(text_path) as f:
        for line in f.readlines():
            texts.append(line)
    return texts


def corpus_perplexity(corpus_path, model):
    texts = read(corpus_path)
    N = sum(len(x.split()) for x in texts)
    corpus_prep = 0
    for text in texts:
        sen_perp = model.perplexity(text)
        sen_perp_normed = sen_perp ** (len(text.split())/N)
        corpus_prep += np.log(sen_perp_normed)

        #print(corpus_prep)
    return corpus_prep

if __name__ == "__main__":
    conllu_test_path = "/home/tsargsyan/davit/arm/data/normalized/test-conllu.txt"
    arpa_test_path = "/home/tsargsyan/davit/arm/data/normalized/test-arpa.txt"
    
    conllu_perps = []
    arpa_perps = []
    n_grams = [2,3,4,5,6,7,8,9,10]

    n_gram_models = os.listdir("lms")
    for n_gram in n_grams:
        model = None
        for model_name in n_gram_models:
            if model_name.startswith("char-{}".format(n_gram)):
                model = kenlm.Model("lms/{}".format(model_name))
                break
        conllu_perps.append(corpus_perplexity(corpus_path=conllu_test_path, model=model))
        arpa_perps.append(corpus_perplexity(corpus_path=arpa_test_path, model=model))
        print(arpa_perps)

    plt.plot(n_grams, conllu_perps, 'g', marker='D', label='conllu')
    plt.plot(n_grams, arpa_perps, 'r', marker='X', label='ARPA')

    plt.xlabel('n-gram')
    plt.ylabel('log(perplexity)')
    plt.legend()

    plt.savefig("ngram-vs-perplexity.png", dpi=200)