def read(text_path):
    texts = []
    with open(text_path) as f:
        for line in f.readlines():
            texts.append(line.strip())
    return texts

def corpus_perplexity(corpus_path, model):
    texts = read(corpus_path)
    N = sum(len(x.split()) for x in texts)

    corpus_perp = 1
    for text in texts:
        sen_perp = model.perplexity(text)
        sen_perp_normed = sen_perp ** (len(text.split())/N)
        corpus_perp *= sen_perp_normed
    return corpus_perp
