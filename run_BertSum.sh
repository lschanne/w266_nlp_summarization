#!/bin/bash

PREV_DIR=$(pwd)
D="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${D}

MODEL_DIR=${D}/models
DATASET=cnn_dailymail
DATA_DIR=${D}/data
DATASET_DIR=${DATA_DIR}/${DATASET}
MAP_DIR=${DATASET_DIR}/BertSum_maps
RAW_DIR=${DATASET_DIR}/BertSum_stories
TOKENIZED_DIR=${DATASET_DIR}/tokenized
LINE_FORMATTED_DIR=${DATASET_DIR}/BertSum_line_formatted
BERT_FORMATTED_DIR=${DATASET_DIR}/BertSum_bert_formatted
LOG_DIR=${DATASET_DIR}/BertSum_logs
BERT_DIR=${DATA_DIR}/pretained_bert
PYTORCH_PRETRAINED_BERT_CACHE="${BERT_DIR}"
BERTSUM_DIR=${D}/BertSum
BERTSUM_SRC=${BERTSUM_DIR}/src
PY_SCRIPT=${BERTSUM_SRC}/preprocess.py
CORE_NLP_DIR=${D}/stanford-corenlp-full-2017-06-09
RESULTS_DIR=${DATASET_DIR}/BertSum_results
FILE_PREFIX=${DATASET}

export PYTORCH_PRETRAINED_BERT_CACHE=${DATA_DIR}/pretrained_bert
for file in `find ${CORE_NLP_DIR} -name "*.jar"`
do
    export CLASSPATH="${CLASSPATH}:`realpath $file`";
done

source venv_w266_final/bin/activate

echo "Tokenize..."
mkdir -p ${TOKENIZED_DIR}
mkdir -p ${LOG_DIR}
python ${PY_SCRIPT} -mode tokenize -raw_path ${RAW_DIR} \
    -save_path ${TOKENIZED_DIR} -map_path ${MAP_DIR} \
    -log_file ${LOG_DIR}/tokenize.log

echo "Format to lines..."
mkdir -p ${LINE_FORMATTED_DIR}
python ${PY_SCRIPT} -mode format_to_lines -raw_path ${TOKENIZED_DIR} \
    -save_path ${LINE_FORMATTED_DIR}/${DATASET} -map_path ${MAP_DIR} \
    -lower -log_file ${LOG_DIR}/format_to_lines.log -n_cpus 8

echo "Format to bert..."
mkdir -p ${BERT_FORMATTED_DIR}
python ${PY_SCRIPT} -mode format_to_bert -raw_path ${LINE_FORMATTED_DIR} \
    -save_path ${BERT_FORMATTED_DIR} -map_path ${MAP_DIR} -oracle_mode greedy \
    -n_cpus 8 -log_file ${LOG_DIR}/format_to_bert.log

N_GPU=1
GPUS=0
BATCH_SIZE=100

echo "Training..."
python ${BERTSUM_SRC}/train.py -mode train -encoder transformer -dropout 0.1 \
    -bert_data_path ${BERT_FORMATTED_DIR}/${DATASET} \
    -model_path ${MODEL_DIR}/bertsum_transformer -lr 2e-3 \
    -save_checkpoint_steps 1000 -batch_size ${BATCH_SIZE} -decay_method noam \
    -train_steps 50000 -accum_count 2 -log_file ${LOG_DIR}/bert_transformer \
    -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 \
    -visible_gpus ${GPUS} -gpu_ranks ${GPUS} -world_size ${N_GPU} \
    -heads 8

echo "Evaluating..."
mkdir -p ${RESULTS_DIR}
python ${BERTSUM_SRC}/train.py -mode validate \
    -bert_data_path ${BERT_FORMATTED_DIR}/${DATASET} \
    -model_path ${MODEL_DIR}/bertsum_transformer  -visible_gpus 3  \
    -visible_gpus ${GPUS} -gpu_ranks ${GPUS} -world_size ${N_GPU} \
    -log_file ${LOG_DIR}/bertsum_transformer_evaluation  \
    -result_path ${RESULTS_DIR}/bertsum_transformer -test_all \
    -batch_size ${BATCH_SIZE} -block_trigram true -temp_dir /tmp \
    -bert_config_path ${BERTSUM_DIR}/bert_config_uncased_base.json

cd ${PREV_DIR}
