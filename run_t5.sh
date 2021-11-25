#!/bin/bash

D="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PreSumm=${D}/PreSumm
cd ${D}

source ${D}/venv_w266_final/bin/activate

OUTPUT_DIR=${D}/data/abs_outputs_1
BERT_DATA_PATH=${PreSumm}/bert_data/bert_data_cnndm_final/cnndm
EXT_MODEL_PATH=${PreSumm}/models/cnndm_ext/cnndm/model_step_45000.pt
LOG_PATH=${D}/data/hybrid_log
BATCH_SIZE=500
# T5_MODEL="t5-large"
T5_MODEL="t5-base"

mkdir -p ${OUTPUT_DIR}

python hybrid_model.py \
    -do_abstraction 1 -train_abs 1 -gen_summaries 1 -do_evaluation 1 \
    -do_extraction 0 -bert_data_path ${BERT_DATA_PATH} \
    -abs_model_path ${OUTPUT_DIR}/abs_model \
    -ext_model_path ${EXT_MODEL_PATH} -batch_size ${BATCH_SIZE} \
    -log_file ${LOG_PATH} -output_dir ${OUTPUT_DIR} \
    -ext_dropout 0.1 -lr 2e-3 -visible_gpus 0 \
    -report_every 500 -save_checkpoint_steps 1000 \
    -train_steps 50000 -accum_count 2 \
    -use_interval true -warmup_steps 10000 -max_pos 512 \
    -abs_epochs 1 -abs_max_input_len 512 -abs_max_output_len 150 \
    -abs_learning_rate 1e-4 -abs_batch_size 2 -abs_num_beams 2 \
    -abs_rep_penalty 2.5 -abs_length_penalty 1.0 -abs_early_stopping 1 \
    -t5_model ${T5_MODEL} -abs_min_output_len 40 -seed 42
