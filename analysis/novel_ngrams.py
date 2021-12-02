'''
Compute "novel" n-grams from abstractive models. Novel here means that it is
an n-gram that apperas in the generated summary and not in the source document.

Most of the code here is copied from PreSumm/src/post_stats.py
'''
from datetime import datetime
import os
import re
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(DIR)
OUTPUT_DIR = os.path.join(BASE, 'data', 'rouge_analysis')
sys.path.extend([
    os.path.join(BASE, 'PreSumm', 'src'),
])

# PreSumm imports
from post_stats import n_grams

def main(n_values=(1, 2, 4)):
    start = datetime.now()
    summaries = {
        name: get_summaries(name) for name in (
            'abstractive_summaries', 'hybrid_summaries', 'target_summaries',
        )
    }
    with open(os.path.join(OUTPUT_DIR, 'articles'), 'r') as f:
        source_lines = [
            re.sub(
                r' +', ' ',
                (source.replace(' ##', ' ').replace('[CLS]', ' ')
                    .replace('[SEP]', ' ').replace('[PAD]', ' ')),
            ).strip()
            for source in f.read().strip().split('\n')
        ]
    
    novel_ngrams = {
        name: {n: [0, 0, 0] for n in n_values}
        for name in summaries.keys()
    }
    for iii, source in enumerate(source_lines):
        if iii % 500 == 0:
            print(
                f'Computing novel ngrams for doc {iii} out of '
                f'{len(source_lines)}; time elapsed: {start - datetime.now()}'
            )
        for n in n_values:
            source_grams = set(n_grams(source.split(), n))
            for name, summary_lines in summaries.items():
                summary_grams = set(n_grams(summary_lines[iii].split(), n))
                novel = summary_grams - (summary_grams & source_grams)
                novel_ngrams[name][n][0] += 1.0 * len(novel)
                novel_ngrams[name][n][0] += len(summary_grams)
                novel_ngrams[name][n][0] += 1.0 * len(novel) / (
                    len(summary_lines[iii].split()) + 1e-6
                )
    
    output = {
        name: {n: values[0] / values[1] for n, values in ngrams.items()}
        for name, ngrams in novel_ngrams.items()
    }
    with open(os.path.join(OUTPUT_DIR, 'novel_ngrams_output'), 'w') as f:
        f.write(str(output))
    print(output)


def get_summaries(name):
    with open(os.path.join(OUTPUT_DIR, name), 'r') as f:
        return [
            re.sub(r' +', ' ', summary.replace('<q>', ' ')).strip()
            for summary in f.read().strip().split('\n')
        ]