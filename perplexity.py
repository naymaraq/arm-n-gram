import kenlm
import sys

binary_file ="ngram-vs-perp/lms/char-5-0|0|0|0|0-q8-b8-a64.binary"
model = kenlm.Model(binary_file)

text_path = sys.argv[1]
with open(text_path) as f:
    text = f.read()
sen_perp = model.perplexity(text)
print(sen_perp)
