#!/bin/bash

# NOTE getting tensorflow gpu setup is a big pain and very system specific, so
# it's not very viable to add to the script here
# if you want to use gpus for training/evaluating the models, and I definitely
# recommend you do, then you'll have to manually set that up yourself.
# This can be done after running this install.sh script

D="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BERT_DIR=${D}/data/pretrained_bert

sudo apt-get update
sudo apt-get install python3 unzip default-jre libxml-parser-perl python3-venv

cd ${D}
python3 -m venv venv_w266_final
# echo "export PYTHONPATH=\"\${PYTHONPATH}:${D}/PreSumm/src\"" > venv_w266_final/bin/postactivate
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

# Install pyrouge
git clone -b master https://github.com/bheinzerling/pyrouge
cd pyrouge
git checkout 08e9cc3
pip install -e .

# Install official rouge script
git clone -b master https://github.com/andersjo/pyrouge.git rouge
cd rouge
git checkout 3b6c415
cd ..
mkdir -p ${D}/pyrouge/rouge/tools/ROUGE-1.5.5/data
pyrouge_set_rouge_path ${D}/pyrouge/rouge/tools/ROUGE-1.5.5/

# Regenerate exceptions DB for pyrouge
cd rouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
python -m pyrouge.test
cd ${D}


export PYTHONPATH="${PYTHONPATH}:${D}/BertSum/src"
python ./preprocess_data.py -do_bertsum 0 -do_t5 1

cd ${PREV_DIR}
