import sys
import sentencepiece as spm


train_txt_path = sys.argv[1]
vocab_size = int(sys.argv[2])
spm.SentencePieceTrainer.train(input=train_txt_path, 
                               model_prefix='arm', 
                               vocab_size=vocab_size,
                               model_type="bpe")