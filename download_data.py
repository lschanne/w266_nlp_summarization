# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/catalog/gigaword

import os

import tensorflow_datasets as tfds

dataset_name = 'gigaword'
DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(DIR, 'data', dataset_name)
DOC_DIR = os.path.join(DATA_DIR, 'raw_documents')
SUMMARY_DIR = os.path.join(DATA_DIR, 'raw_summaries')
MAP_DIR = os.path.join(DATA_DIR, 'bertsum_maps')

for dir_ in (DOC_DIR, SUMMARY_DIR, MAP_DIR):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

builder = tfds.builder(dataset_name)
builder.download_and_prepare(download_dir=DATA_DIR)

data = builder.as_dataset()

for key, subset in data.items():
    if key == 'validation':
        key = 'valid'
    
    files = []
    for idx, row in enumerate(subset):
        unhashed_fh = f'{key}{idx}'
        hashed_fh = hashhex(unhashed_fh)
        files.append(f'{unhashed_fh}\n')
        for type_, d in (
            ('document', DOC_DIR),
            ('summary', SUMMARY_DIR),
        ):
            text = bytes.decode(row[type_].numpy())
            fh = os.path.join(d, f'{hashed_fh}.story')
            with open(fh, 'w') as f:
                f.write(text)

    with open(os.path.join(MAP_DIR, f'mapping_{key}.txt'), 'w') as f:
        f.writelines(files)

