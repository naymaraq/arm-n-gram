import os, sys
import subprocess
from dataclasses import dataclass
import sentencepiece
import yaml
from copy import copy


import kenlm_utils

TOKEN_OFFSET = 100
CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512

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

    def prepare_dataset(self, encoding_level, tokenizer_path=None):
        
        assert encoding_level in ["char", "subword"]
        dataset = kenlm_utils.read_train_file(self.train_file, lowercase=self.do_lowercase)
        encoded_train_file = f"{self.kenlm_models_folder}/dataset.tmp.txt"
        discount_arg = ""
        if encoding_level == "subword":
            tokenizer = load_tokenizer(tokenizer_path)
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
        trainer_config, ngram_config, encoding_level, tokenizer_path = args
        trainer = KenLMTrainer(**trainer_config)
        trainer.prepare_dataset(encoding_level, tokenizer_path)
        for ngram, prune, tmp_folder, q, b, a in ngram_config:
            model_path = trainer.fit(ngram, prune, tmp_folder, q, b, a)
            print(f"Finished constructing {model_path}")
    except Exception as e:
        print(e)


def read_yaml(yaml_path):
    with open(yaml_path, "r") as stream:
        data = yaml.safe_load(stream)
    return data

if __name__ == "__main__":

    config_path = sys.argv[1]
    config = read_yaml(config_path)

    N, P, Q = int(config["N"]), int(config["P"]), int(config["Q"])
    kenlm_bin_path = config["kenlm_bin_path"]
    where_to_store = config["where_to_store"]
    custom_tmp_folder = config["custom_tmp_folder"]

    datasets = {"lms": [config["txt_path"]]}
    base_config = KenLMTrainerConfig(
        kenlm_bin_path=kenlm_bin_path
    )

    key = "lms"
    dataset = datasets[key]
    cfg = copy(base_config)
    cfg.set_train_file(dataset)
    os.makedirs(os.path.join(where_to_store, key), exist_ok=True)
    cfg.set_models_folder(os.path.join(where_to_store, key))
    dataset_config = cfg.todict()
    
    ngram_config=[]
    ngram_config.append((N, "|".join([f"{P}"]*N), custom_tmp_folder, Q, Q, 64))

    encoding_level = config["encoding_level"]
    tokenizer_path = None
    if encoding_level == "subword":
        tokenizer_path = config["tokenizer_path"]
    train_one((dataset_config, ngram_config, encoding_level, tokenizer_path))
