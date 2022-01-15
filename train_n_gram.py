import os
import subprocess
from dataclasses import dataclass
import sys
from multiprocessing import Pool
import progressbar


import kenlm_utils
from copy import copy

TOKEN_OFFSET = 100
CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512

import sentencepiece
def load_tokenizer(model_path):
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.Load(model_path)
    return tokenizer


@dataclass
class KenLMTrainer:
    """KenLM Trainer"""
    train_file: str
    kenlm_bin_path: str
    kenlm_models_folder: str
    do_lowercase: bool=False
    memory: str="25%"

    def prepare_dataset(self):
        
        encoding_level = "char"
        dataset = kenlm_utils.read_train_file(self.train_file, lowercase=self.do_lowercase)

        encoded_train_file = f"{self.kenlm_models_folder}/dataset.tmp.txt"
        discount_arg = ""
        if encoding_level == "subword":
            tokenizer = load_tokenizer("tokenizer.model")
            kenlm_utils.tokenize_text(
                dataset,
                tokenizer,
                path=encoded_train_file,
                chunk_size=CHUNK_SIZE,
                buffer_size=CHUNK_BUFFER_SIZE,
                token_offset=TOKEN_OFFSET,
            )
            # --discount_fallback is needed for training KenLM for BPE-based models
            discount_arg = "--discount_fallback"
        else:
            with open(encoded_train_file, 'w', encoding='utf-8') as f:
                for line in dataset:
                    f.write(f"{line}\n")
        del dataset

        self.encoded_train_file = encoded_train_file
        self.encoding_level = encoding_level
        self.discount_arg = discount_arg

    def fit(self, n_gram, arpa_prune, tmp_folder_path=None, q=None, b=None, a=None):
        arpa_file = f"{self.encoding_level}-{n_gram}-{arpa_prune}.tmp.arpa"
        arpa_file = os.path.join(self.kenlm_models_folder, arpa_file)

        additional_args = ""
        if q is not None:
            additional_args += " -q %s"%q
        if b is not None:
            additional_args += " -b %s"%b
        if a is not None:
            additional_args += " -a %s"%a
        additional_args_in_name = additional_args.replace(" ", "")
        kenlm_model_file = f"{self.encoding_level}-{n_gram}-{arpa_prune}" + additional_args_in_name + ".binary"
        kenlm_model_file = os.path.join(self.kenlm_models_folder, kenlm_model_file)

        if additional_args != "":
            additional_args = "trie" + additional_args

        if not os.path.exists(kenlm_model_file):
            kenlm_args = [
                os.path.join(self.kenlm_bin_path, 'lmplz'),
                "-o",
                f"{n_gram}",
                "--text",
                self.encoded_train_file,
                "--arpa",
                arpa_file,
                "--memory",
                f"{self.memory}",
                self.discount_arg
            ]
            if set(arpa_prune.split("|")) != {"0"}:
                kenlm_args += ["--prune",
                               *arpa_prune.split("|")
                              ]

            if tmp_folder_path != None:
                kenlm_args += ["-T", tmp_folder_path]

            ret = subprocess.run(kenlm_args, stdout=sys.stdout, stderr=sys.stderr)
            if ret.returncode != 0:
                raise RuntimeError("Training KenLM was not successful!")

            kenlm_args = [
                os.path.join(self.kenlm_bin_path, "build_binary"),
                *additional_args.split(),
                arpa_file,
                kenlm_model_file,
            ]
            ret = subprocess.run(kenlm_args,  stdout=sys.stdout, stderr=sys.stderr)

            if ret.returncode != 0:
                raise RuntimeError("Training KenLM was not successful!")

            os.remove(arpa_file)

        return kenlm_model_file


@dataclass
class KenLMTrainerConfig:
    kenlm_bin_path: str
    do_lowercase: bool=False
    memory: str="25%"

    def set_train_file(self, train_files):
        self.train_file = ','.join(train_files)

    def set_models_folder(self, folder):
        self.kenlm_models_folder = folder

    def todict(self):
        return {
                "kenlm_models_folder": self.kenlm_models_folder,
                "kenlm_bin_path": self.kenlm_bin_path,
                "do_lowercase": self.do_lowercase,
                "memory": self.memory,
                "train_file": self.train_file
               }
    
    def __repr__(self):
        return str({
                    "kenlm_models_folder": self.kenlm_models_folder,
                    "datasets": self.train_file.split(",")})

def train_one(args):
    try:
        trainer_config, ngram_config = args
        trainer = KenLMTrainer(**trainer_config)
        trainer.prepare_dataset()
        for ngram, prune, tmp_folder, q, b, a in ngram_config:
            model_path = trainer.fit(ngram, prune, tmp_folder, q, b, a)
            print(f"Finished constructing {model_path}")
    except Exception as e:
        print(e)


if __name__ == "__main__":

    where_to_store = sys.argv[1]
    custom_tmp_folder = None
    if len(sys.argv[1:]) > 1:
        custom_tmp_folder = sys.argv[2]

    kenlm_bin_path = "/home/tsargsyan/davit/arm/decoders/kenlm/build/bin"

    #dataset0 = ["/home/tsargsyan/davit/arm/data/normalized/arm-wiki.txt",
    #           "/home/tsargsyan/davit/arm/data/normalized/train-arpa.txt",
    #           "/home/tsargsyan/davit/arm/data/normalized/train-conllu.txt"]
    #datasets = {"lms": dataset0}
    base_dataset = [None]
    base_config = KenLMTrainerConfig(
        kenlm_bin_path=kenlm_bin_path
    )

    dataset_configs=[]
    for key, dataset in datasets.items():
        cfg = copy(base_config)
        cfg.set_train_file(dataset)
        os.makedirs(os.path.join(where_to_store, key), exist_ok=True)
        cfg.set_models_folder(os.path.join(where_to_store, key))
        dataset_configs.append(cfg.todict())
    
    ngram_config=[]
    for n in [2,3,4,5,6,7,8,9,10]:
        ngram_config.append((n, "|".join(["0"]*n), custom_tmp_folder, 8, 8, 64))
        # ngram_config.append((n, "|".join(["0"]*n), custom_tmp_folder, None, None, None))
        #prune = [50 for i in range(n)]
        #ngram_config.append((n, "|".join(map(str, prune)), custom_tmp_folder, 8, 8, 64))

    args = []
    for dataset_config in dataset_configs:
        args.append((dataset_config, ngram_config))
    
    pool = Pool()
    bar = progressbar.ProgressBar(max_value=len(args))
    for i, _ in enumerate(pool.imap_unordered(train_one, args)):
        bar.update(i)
    bar.update(len(args))
    pool.close()
    pool.join()
