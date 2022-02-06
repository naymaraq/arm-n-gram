# arm-n-gram
N-gram Language Models for Armenian

# Installation
  * Install KenLM (https://kheafield.com/code/kenlm/)
  ```(bash)
  sudo apt-get update
  sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
  wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
  mkdir kenlm/build
  cd kenlm/build
  cmake ..
  make -j2
  ```
  * pip install -r requirements.txt

# To Build Tokenizer
  ```(bash) 
    python train_tokenizer.py train_txt_path vocab_size
  ```

# To Build Language Model
  * change parameters in ```conf/config.yaml ```
  ```(bash)
     python train_n_gram.py conf/config.yaml
  ```
