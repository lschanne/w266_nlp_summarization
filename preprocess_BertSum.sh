#!/bin/bash

PREV_DIR=$(pwd)
D="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#git clone -b master git@github.com:nlpyang/BertSum.git --single-branch
#cd BertSum && git checkout 05f8c63 && cd ..
#wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
#unzip stanford-corenlp-full-2017-06-09.zip
CORE_NLP_DIR=${D}/stanford-corenlp-full-2017-06-09
cd ${CORE_NLP_DIR}
for file in `find . -name "*.jar"`
do
    export CLASSPATH="${CLASSPATH}:`realpath $file`";
done
echo ${CLASSPATH}

cd ${D}
source venv_w266_final/bin/activate

BERTSUM_DIR=${D}/BertSum
PY_SCRIPT=${BERTSUM_DIR}/src/preprocess.py

DATA_DIR=${D}/data/gigaword
MAP_DIR=${BERTSUM_DIR}/bertsum_maps
RAW_DIR=${DATA_DIR}/raw_documents
TOKENIZED_DIR=${DATA_DIR}/tokenized
LINE_FORMATTED_DIR=${DATA_DIR}/line_formatted
BERT_FORMATTED_DIR=${DATA_DIR}/bert_formatted
LOG_DIR=${DATA_DIR}/logs

echo "Tokenize..."
mkdir -p ${TOKENIZED_DIR}
mkdir -p ${LOG_DIR}
python ${PY_SCRIPT} -mode tokenize -raw_path ${RAW_DIR} -save_path ${TOKENIZED_DIR} -map_path ${MAP_DIR} -log_file ${LOG_DIR}/tokenize.log

echo "Format to lines..."
mkdir -p ${LINE_FORMATTED_DIR}
python ${PY_SCRIPT} -mode format_to_lines -raw_path ${TOKENIZED_DIR} -save_path ${LINE_FORMATTED_DIR} -map_path ${MAP_DIR} -lower  -log_file ${LOG_DIR}/format_to_lines.log
#python ${PY_SCRIPT} -mode format_to_lines -raw_path ${RAW_DIR} -save_path ${LINE_FORMATTED_DIR} -map_path ${MAP_DIR} -lower  -log_file ${LOG_DIR}/format_to_lines.log

echo "Format to bert..."
mkdir -p ${BERT_FORMATTED_DIR}
python ${PY_SCRIPT} -mode format_to_bert -raw_path ${LINE_FORMATTED_DIR} -save_path ${BERT_FORMATTED_DIR} -map_path ${MAP_DIR} -oracle_mode greedy -n_cpus 4 -log_file ${LOG_DIR}/format_to_bert.log

# train
#python train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 0,1,2  -gpu_ranks 0,1,2 -world_size 3 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8


# evaluate
#python train.py -mode validate -bert_data_path ../bert_data/cnndm -model_path MODEL_PATH  -visible_gpus 0  -gpu_ranks 0 -batch_size 30000  -log_file LOG_FILE  -result_path RESULT_PATH -test_all -block_trigram true


cd ${PREV_DIR}
