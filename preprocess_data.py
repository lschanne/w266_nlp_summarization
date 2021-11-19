# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/catalog/gigaword

import argparse
import os

import numpy as np
import tensorflow_datasets as tfds

# BertSum imports
from prepro.data_builder import hashhex
from train import str2bool

class PreProcesser:
    KEY_MAP = {
        'gigaword': {
            'document': 'document',
            'summary': 'summary',
        },
        'cnn_dailymail': {
            'document': 'article',
            'summary': 'highlights',
        },
    }

    def __init__(
        self,
        dataset_name='cnn_dailymail',
        do_bertsum=True,
        do_t5=True,
    ):
        self.dataset_name = dataset_name
        self.doc_key = self.KEY_MAP[dataset_name]['document']
        self.summary_key = self.KEY_MAP[dataset_name]['summary']
        self.do_bertsum = do_bertsum
        self.do_t5 = do_t5


        self.DIR = os.path.abspath(os.path.dirname(__file__))
        self.DATA_DIR = os.path.join(self.DIR, 'data', dataset_name)
        self.DOC_DIR = os.path.join(self.DATA_DIR, 'raw_documents')
        self.SUMMARY_DIR = os.path.join(self.DATA_DIR, 'raw_summaries')
        self.MAP_DIR = os.path.join(self.DATA_DIR, 'BertSum_maps')
        self.STORY_DIR = os.path.join(self.DATA_DIR, 'BertSum_stories')

    def main(self):
        self.makedirs()
        data, info = tfds.load(self.dataset_name, split='train', with_info=True)
        if self.do_bertsum:
            self.process_for_bertsum(data)
        if self.do_t5:
            self.process_for_t5(data, info)

    def makedirs(self):
        for dir_ in (
            self.DOC_DIR, self.SUMMARY_DIR, self.MAP_DIR, self.STORY_DIR
        ):
            if not os.path.exists(dir_):
                os.makedirs(dir_)

    def process_for_bertsum(self, data):
        for key, subset in data.items():
            if key == 'validation':
                key = 'valid'
            
            files = []
            for idx, row in enumerate(subset):
                unhashed_fh = f'{key}{idx}'
                hashed_fh = hashhex(unhashed_fh)
                files.append(f'{unhashed_fh}\n')
                for type_, d in (
                    (self.doc_key, self.DOC_DIR),
                    (self.summary_key, self.SUMMARY_DIR),
                ):
                    text = bytes.decode(row[type_].numpy())
                    fh = os.path.join(d, unhashed_fh)
                    with open(fh, 'w') as f:
                        f.write(text)
                
                text = '\n'.join((
                    bytes.decode(row[self.doc_key].numpy()),
                    '\n@highlight\n',
                    bytes.decode(row[self.summary_key].numpy()),
                ))
                fh = os.path.join(self.STORY_DIR, f'{hashed_fh}.story')
                with open(fh, 'w') as f:
                    f.write(text)

            with open(
                os.path.join(self.MAP_DIR, f'mapping_{key}.txt'), 'w'
            ) as f:
                f.writelines(files)


    def process_for_t5(self, data, info):
        df = tfds.as_dataframe(data, info)

        # convert from bytes to string lists
        str_df = (
            df.select_dtypes([np.object]).stack().str.decode('utf-8').unstack()
        )
        for col in str_df:
            df[col] = str_df[col]

        df.dropna(inplace=True)

        # remove special characters and lowercase
        for key in (self.doc_key, self.summary_key):
            df[key] = df[key].str.replace('[^A-Za-z\s]+', '').str.lower()

        df.to_csv(os.path.join(self.DATA_DIR, f'{self.dataset_name}.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-do_bertsum', type=str2bool, nargs='?', const=True, default=True,
    )
    parser.add_argument(
        '-do_bertsum', type=str2bool, nargs='?', const=True, default=True,
    )
    parser.add_argument(
        '-dataset_name', default='cnn_dailymail',
    )
    args = parser.parse_args()
    PreProcesser(
        dataset_name=args.dataset_name,
        do_bertsum=args.do_bertsum,
        do_t5=args.do_t5,
    ).main()
