# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/catalog/gigaword

import os

import tensorflow_datasets as tfds

from prepro.data_builder import hashhex

dataset_name = 'cnn_dailymail'
DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(DIR, 'data', dataset_name)
DOC_DIR = os.path.join(DATA_DIR, 'raw_documents')
SUMMARY_DIR = os.path.join(DATA_DIR, 'raw_summaries')
MAP_DIR = os.path.join(DATA_DIR, 'BertSum_maps')
STORY_DIR = os.path.join(DATA_DIR, 'BertSum_stories')

for dir_ in (DOC_DIR, SUMMARY_DIR, MAP_DIR, STORY_DIR):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

data = tfds.load(dataset_name)

key_map = {
    'gigaword': {
        'document': 'document',
        'summary': 'summary',
    },
    'cnn_dailymail': {
        'document': 'article',
        'summary': 'highlights',
    },
}

doc_key = key_map[dataset_name]['document']
summary_key = key_map[dataset_name]['summary']
for key, subset in data.items():
    if key == 'validation':
        key = 'valid'
    
    files = []
    for idx, row in enumerate(subset):
        unhashed_fh = f'{key}{idx}'
        hashed_fh = hashhex(unhashed_fh)
        files.append(f'{unhashed_fh}\n')
        for type_, d in (
            (doc_key, DOC_DIR),
            (summary_key, SUMMARY_DIR),
        ):
            text = bytes.decode(row[type_].numpy())
            fh = os.path.join(d, unhashed_fh)
            with open(fh, 'w') as f:
                f.write(text)
        
        text = '\n'.join((
            bytes.decode(row[doc_key].numpy()),
            '\n@highlight\n',
            bytes.decode(row[summary_key].numpy()),
        ))
        fh = os.path.join(STORY_DIR, f'{hashed_fh}.story')
        with open(fh, 'w') as f:
            f.write(text)

    with open(os.path.join(MAP_DIR, f'mapping_{key}.txt'), 'w') as f:
        f.writelines(files)

