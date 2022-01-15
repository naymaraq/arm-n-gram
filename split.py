def read(text_path):
    texts = []
    with open(text_path) as f:
        for line in f.readlines():
            texts.append(line)
    return texts

def segment_text(text, n):
    text = text.split()
    return [' '.join(text[i:i+n]) for i in range(0,len(text),n)]


def write_text(text, path):
    with open(path, "w") as f:
        f.write(text)

import sys
text = read(sys.argv[1])[0]
segs = segment_text(text,10**6)

for i in range(1,len(segs)+1):
    write_text(" ".join(segs[:i]),f"datasize-vs-perp/parts/{i}M.txt")
