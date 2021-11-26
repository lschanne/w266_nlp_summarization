'''
Get the rouge score of each document with the generated summary from the
extractive, abstractive, and hybrid models.
'''

from collections import defaultdict
from datetime import datetime
import os
import pandas as pd
import shutil
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(DIR)
DATA_DIR = os.path.join(BASE, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'rouge_analysis')
sys.path.extend([
    os.path.join(BASE, 'PreSumm', 'src'),
])

# PreSumm imports
from others import pyrouge

def main():
    copy_files()
    for model in ('extractive', 'abstractive', 'hybrid'):
        compute_rouge(model)

def copy_files():
    print('Copying files...')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    shutil.copy(
        os.path.join(DATA_DIR, 'hybrid_outputs_1', 'valid', 'target'),
        os.path.join(OUTPUT_DIR, 'target_summaries')
    )
    shutil.copy(
        os.path.join(DATA_DIR, 'hybrid_outputs_1', 'valid', 'articles'),
        os.path.join(OUTPUT_DIR, 'extractive_summaries')
    )
    shutil.copy(
        os.path.join(DATA_DIR, 'hybrid_outputs_1', 'valid', 'output'),
        os.path.join(OUTPUT_DIR, 'hybrid_summaries')
    )
    shutil.copy(
        os.path.join(DATA_DIR, 'abs_outputs_1', 'valid', 'output'),
        os.path.join(OUTPUT_DIR, 'abstractive_summaries')
    )
    shutil.copy(
        os.path.join(DATA_DIR, 'abs_outputs_1', 'valid', 'articles'),
        os.path.join(OUTPUT_DIR, 'articles')
    )

def compute_rouge(model):
    print(f'Computing rouge for {model} model...')
    start_time = datetime.now()
    with open(os.path.join(OUTPUT_DIR, f'{model}_summaries'), 'r') as f:
        summaries = f.readlines()
    with open(os.path.join(OUTPUT_DIR, 'target_summaries'), 'r') as f:
        targets = f.readlines()
    
    n_docs = len(summaries)
    data = defaultdict(lambda: [])
    for doc_id, (summary, target) in enumerate(zip(summaries, targets)):
        if doc_id % 500 == 0:
            print(
                f'Computing rouge for {model} doc {doc_id} out of {n_docs}; '
                f'Time elapsed: {datetime.now() - start_time}'
            )
        rouge_dir = os.path.join(OUTPUT_DIR, 'rouge_files')
        temp_dir = os.path.join(OUTPUT_DIR, 'rouge_temp')
        for d in (rouge_dir, temp_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        with open(os.path.join(rouge_dir, f't_{doc_id}'), 'w') as f:
            f.write(target)
        with open(os.path.join(rouge_dir, f'p_{doc_id}'), 'w') as f:
            f.write(summary)

        rouge = pyrouge.Rouge155(temp_dir=temp_dir)
        rouge.model_dir = rouge_dir
        rouge.system_dir = rouge_dir
        rouge.model_filename_pattern = 'p_#ID#'
        rouge.system_filename_pattern = r't_(\d+)'
        scores = rouge.output_to_dict(rouge.convert_and_evaluate())
        scores['doc_id'] = doc_id
        for key, value in scores.items():
            data[key].append(value)
    
    pd.DataFrame(data).to_csv(os.path.join(OUTPUT_DIR, f'{model}_rouge.csv'))

    
if __name__ == '__main__':
    main()