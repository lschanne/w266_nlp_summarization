#!/bin/bash


D="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PreSumm=${D}/PreSumm
cd ${D}

source ${D}/venv_w266_final/bin/activate

git clone git@github.com:lschanne/PreSumm.git

cd ${PreSumm}/bert_data
pip install gdown
gdown https://drive.google.com/uc?id=1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI
unzip bert_data_cnndm_final.zip

cd ${PreSumm}/src/data
BERT_DATA_PATH=${PreSumm}/bert_data/bert_data_cnndm_final/cnndm
MODEL_PATH=${PreSumm}/models/cnndm_ext/cnndm
LOG_PATH=${PreSumm}/logs
BATCH_SIZE=500

python train.py -task ext -mode train -bert_data_path ${BERT_DATA_PATH} \
    -model_path ${MODEL_PATH} -batch_size ${BATCH_SIZE} \
    -log_file ${LOG_PATH}/ext_bert_cnndm \
    -ext_dropout 0.1 -lr 2e-3 -visible_gpus 0 \
    -report_every 50 -save_checkpoint_steps 1000 \
    -train_steps 50000 -accum_count 2 \
    -use_interval true -warmup_steps 10000 -max_pos 512

python train.py -task ext -mode validate -batch_size ${BATCH_SIZE} \
    -test_batch_size 500 -bert_data_path ${BERT_DATA_PATH} -test_all \
    -log_file ${LOG_PATH}/val_ext_bert_cnndm -model_path ${MODEL_PATH} \
    -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 \
    -max_length 200 -alpha 0.95 -min_length 50 \
    -result_path ${LOG_PATH}/ext_bert_cnndm_results
