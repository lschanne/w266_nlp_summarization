#!/bin/bash
D="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BERT_DIR=${D}/data/pretrained_bert

sudo apt-get update
sudo apt-get install python3 unzip default-jre
cd ${D}
python3 -m venv venv_w266_final
source venv_w266_final/bin/activate

pip install --upgrade pip
pip install wheel
pip install --upgrade setuptools
pip install -r requirements.txt

git clone git@github.com:lschanne/BertSum.git
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
unzip stanford-corenlp-full-2017-06-09.zip
rm stanford-corenlp-full-2017-06-09.zip

mkdir -p ${BERT_DIR}
cd ${BERT_DIR}
wget -q -O- https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz | tar -xvz
cd ${D}


export PYTHONPATH="${PYTHONPATH}:${D}/BertSum/src"
python ./download_data.py

cd ${PREV_DIR}
