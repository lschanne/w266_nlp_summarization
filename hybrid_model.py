'''
Use the extractive summaries from BERTSum as the input documents to t5
abstractive summarization.
'''

import argparse
from datetime import datetime
import gc
import glob
import numpy as np
import os
import pandas as pd
from pyrouge import Rouge155
import re
import shutil
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(DIR, 'PreSumm', 'src'),
])

# PreSumm imports
from cal_rouge import rouge_results_to_str
from models.data_loader import Dataloader, load_dataset
from models.model_builder import ExtSummarizer
from models.reporter_ext import Statistics
from train import str2bool
from train_extractive import model_flags as PreSumm_model_flags

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


class HybridModel:
    def __init__(self):
        pass

    @classmethod
    def main(cls, args):
        args = cls._update_args(args)
        if args.do_extraction:
            cls.get_extractive_summaries(args)

        torch.backends.cudnn.deterministic = True
        if args.train_abs:
            cls.train_abstractive_stage(args)

        if args.gen_summaries:
            cls.generate_abstractive_summaries(args)

        if args.do_evaluation:
            cls.evaluate(args)

    @classmethod
    def get_extractive_summaries(cls, args):
        device = args.device
        checkpoint = torch.load(
            args.ext_model_path, map_location=lambda storage, loc: storage
        )
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in PreSumm_model_flags):
                setattr(args, k, opt[k])

        model = ExtSummarizer(args, device, checkpoint)
        model.eval()

        stats = Statistics()
        loss_function = torch.nn.BCELoss(reduction='none')

        for corpus in ('train', 'valid', 'test'):
            n_candidates = 1
            if corpus == 'train':
                n_candidates = 3
            data_iter = Dataloader(
                args,
                load_dataset(args, corpus, shuffle=False),
                args.test_batch_size,
                device,
                shuffle=False,
                is_test=True,
            )
            gold_summaries = []
            prediction_summaries = []
            start_time = datetime.now()
            with torch.no_grad():
                for batch_number, batch in enumerate(data_iter, 1):
                    if batch_number % args.report_every == 0:
                        delta = datetime.now() - start_time
                        print(
                            f'Generating extractive sumamries for batch '
                            f'{batch_number}; time elapsed: {delta}'
                        )

                    src = batch.src
                    labels = batch.src_sent_labels
                    segs = batch.segs
                    clss = batch.clss
                    mask = batch.mask_src
                    mask_cls = batch.mask_cls

                    sent_scores, mask = model(
                        src, segs, clss, mask, mask_cls,
                    )

                    loss = loss_function(sent_scores, labels.float())
                    loss = (loss * mask.float()).sum()
                    batch_stats = Statistics(
                        float(loss.cpu().data.numpy()), len(labels),
                    )
                    stats.update(batch_stats)

                    sent_scores = sent_scores + mask.float()
                    sent_scores = sent_scores.cpu().data.numpy()
                    selected_ids = np.argsort(-sent_scores, 1)
                    for i, idx in enumerate(selected_ids):
                        _pred = []
                        if (len(batch.src_str[i]) == 0):
                            continue
                        for j in idx[:len(batch.src_str[i])]:
                            if (j >= len(batch.src_str[i])):
                                continue
                            candidate = batch.src_str[i][j].strip()
                            if (args.block_trigram):
                                if (not cls._block_tri(candidate, _pred)):
                                    _pred.append(candidate)
                            else:
                                _pred.append(candidate)

                            if (
                                (not args.recall_eval) and
                                len(_pred) == n_candidates
                            ):
                                break

                        prediction_summaries.append('<q>'.join(_pred))
                        gold_summaries.append(batch.tgt_str[i])

            summary_dir = os.path.join(args.output_dir, corpus)
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            for key, summaries in (
                ('extractive', prediction_summaries),
                ('target', gold_summaries),
            ):
                with open(os.path.join(summary_dir, key), 'w') as f:
                    f.write('\n'.join(x.strip() for x in summaries))

    @classmethod
    def train_abstractive_stage(cls, args):
        original_device = args.device
        # setattr(args, 'device', 'cpu') # having trouble with gpu memory
        device = args.device

        torch.manual_seed(args.seed) 
        np.random.seed(args.seed)
        start_time = datetime.now()

        train_data = cls._get_data_for_abstraction(args, corpus='train')

        if not os.path.exists(args.abs_model_path):
            os.makedirs(args.abs_model_path)

        model, start_epoch, batch_idx = cls._get_abs_model(args, train=True)

        # optimizer = torch.optim.Adam(
        #     params=model.parameters(), lr=args.abs_learning_rate,
        # )
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=args.abs_learning_rate,
        )
        model.train()

        shuffled_index = np.array(train_data.index)
        total_batches = len(shuffled_index) // args.abs_batch_size
        total_batches += (len(shuffled_index) % args.abs_batch_size) > 0
        for epoch in range(1, start_epoch + 1):
            np.random.shuffle(shuffled_index)
            print(
                f'Found a model for epoch {start_epoch}. '
                f'Skipping epoch {epoch}'
            )
        for epoch in range(start_epoch + 1, args.abs_epochs + 1):
            np.random.shuffle(shuffled_index)
            if batch_idx > 0:
                print(
                    f'Found a model for batch {batch_idx}. '
                    f'Skipping previous batches.'
                )
            start_idx = batch_idx * args.abs_batch_size
            while start_idx < len(shuffled_index):
                batch = train_data.loc[
                    shuffled_index[start_idx:start_idx + args.abs_batch_size]
                ]
                start_idx += args.abs_batch_size
                batch_idx += 1

                optimizer.zero_grad()
                model.zero_grad()
                free_memory()

                loss = model(
                    **cls._get_abs_model_inputs(batch, args, train=True)
                )[0]

                free_memory()

                loss.backward()
                optimizer.step()

                del loss
                free_memory()

                if batch_idx % args.report_every == 0:
                    delta = datetime.now() - start_time
                    print(
                        f'Epoch: {epoch} / {args.abs_epochs}, '
                        f'Batch: {batch_idx} / {total_batches}, '
                        f'Time elapsed: {delta}'
                    )
                if batch_idx % args.save_checkpoint_steps == 0:
                    torch.save(
                        model,
                        os.path.join(
                            args.abs_model_path,
                            f'epoch_{epoch}_batch_{batch_idx}_model.pt'
                        ),
                    )
                    old_fh = os.path.join(
                        args.abs_model_path,
                        f'epoch_{epoch}_batch_'
                        f'{batch_idx - args.save_checkpoint_steps}_model.pt'
                    )
                    if os.path.exists(old_fh):
                        os.remove(old_fh)

            torch.save(
                model,
                os.path.join(
                    args.abs_model_path, f'epoch_{epoch}_model.pt'
                ),
            )
        
        setattr(args, 'device', original_device)
        
    @classmethod
    def generate_abstractive_summaries(cls, args):
        start_time = datetime.now()
        model = cls._get_abs_model(args)[0]
        tokenizer = args.tokenizer
        for corpus in ('valid', 'test'):
            data = cls._get_data_for_abstraction(args, corpus=corpus)
            index = np.array(data.index)
            start_idx = 0
            batch_number = 0
            predictions = []
            while start_idx < len(index):
                batch_number += 1
                batch_idx = index[
                    start_idx:start_idx + args.abs_batch_size
                ]
                batch = data.loc[batch_idx]
                start_idx += args.abs_batch_size
                if batch_number % args.report_every == 0:
                    delta = datetime.now() - start_time
                    print(
                        f'Generating Summaries for {corpus} Batch: '
                        f'{batch_number}, Time elapsed: {delta}'
                    )   

                generated_ids = model.generate(
                    max_length=args.abs_max_output_len,
                    min_length=args.abs_min_output_len,
                    num_beams=args.abs_num_beams,
                    repetition_penalty=args.abs_rep_penalty,
                    length_penalty=args.abs_length_penalty,
                    early_stopping=args.abs_early_stopping,
                    **cls._get_abs_model_inputs(batch, args)
                )

                predictions.extend(
                    tokenizer.decode(
                        g,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ) for g in generated_ids
                )
                del generated_ids
                free_memory()

            data['predictions'] = predictions
            predictions = []
            for _, this_data in data.groupby('doc_id'):
                predictions.append('<q>'.join(this_data['predictions']))

            summary_dir = os.path.join(args.output_dir, corpus)
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            with open(os.path.join(summary_dir, 'hybrid'), 'w') as f:
                f.writelines(p + '\n' for p in predictions)

    @classmethod
    def evaluate(cls, args):
        for corpus in ('valid', 'test'):
            summary_dir = os.path.join(args.output_dir, corpus)
            with open(os.path.join(summary_dir, 'target'), 'r') as f:
                targets = f.readlines()
            with open(os.path.join(summary_dir, 'hybrid'), 'r') as f:
                predictions = f.readlines()
            
            rouge_dir = os.path.join(args.output_dir, corpus, 'rouge_files')
            if not os.path.exists(rouge_dir):
                os.makedirs(rouge_dir)

            idx = 0
            for target, prediction in zip(targets, predictions):
                for t in target.split('<q>'):
                    for p in prediction.split('<q>'):
                        idx += 1
                        with open(
                            os.path.join(rouge_dir, f'target_{idx}'), 'w'
                        ) as f:
                            f.write(t)
                        with open(
                            os.path.join(rouge_dir, f'prediction_{idx}'), 'w'
                        ) as f:
                            f.write(p)
            
            rouge = Rouge155()
            rouge.model_dir = rouge_dir
            rouge.system_dir = rouge_dir
            rouge.model_filename_pattern = 'prediction_#ID#'
            rouge.system_filename_pattern = r'target_(\d+)'
            scores = rouge_results_to_str(
                rouge.output_to_dict(
                    rouge.convert_and_evaluate()
                )
            )
            shutil.rmtree(rouge_dir)

            print(f'{corpus} rouge: {scores}')
            with open(os.path.join(summary_dir, 'hybrid_rouge'), 'w') as f:
                f.write(scores)
            

    @classmethod
    def _get_data_for_abstraction(cls, args, corpus):
        summary_dir = os.path.join(args.output_dir, corpus)
        data_fh = os.path.join(summary_dir, 'abstractive_inputs.csv')
        if os.path.exists(data_fh):
            return pd.read_csv(data_fh)

        with open(os.path.join(summary_dir, 'extractive'), 'r') as f:
            # in this case 'article' is the extractive summary
            articles = f.readlines()

        with open(os.path.join(summary_dir, 'target'), 'r') as f:
            highlights = f.readlines()

        if corpus == 'train':
            flat_article = []
            flat_highlight = []
            doc_id = []
            start_time = datetime.now()
            for idx, (article, highlight) in enumerate(
                zip(articles, highlights)
            ):
                if idx % args.report_every == 0:
                    delta = datetime.now() - start_time
                    print(
                        f'Flattening train data for doc {idx}, '
                        f'Time elapsed: {delta}'
                    )
                article = article.split('<q>')
                highlight = highlight.split('<q>')
                doc_id.extend(np.tile(idx, len(article) * len(highlight)))
                flat_article.extend(
                    np.tile(article, len(highlight))
                )
                flat_highlight.extend(
                    np.repeat(highlight, len(article))
                )
            articles = flat_article
            highlights = flat_highlight
        else:
            doc_id = range(len(articles))

        data = pd.DataFrame({
            'doc_id': doc_id,
            'article': articles,
            'highlights': highlights,
        })
        for col in ('article', 'highlights'):
            data[col] = data[col].apply(cls._clean_col)
        for col in ('article', 'highlights'):
            data = data[data[col].apply(len) > 0]
        data.dropna(inplace=True)
        data.to_csv(data_fh)
        return data

    @staticmethod
    def _clean_col(x):
        return re.sub('\s+', ' ', re.sub('[^A-Za-z\s]+', '', x.strip())).lower()

    @staticmethod
    def _update_args(args):
        if args.abs_model_path is None:
            setattr(
                args,
                'abs_model_path',
                os.path.join(args.output_dir, 'abs_model'),
            )
        setattr(args, 'tokenizer', T5Tokenizer.from_pretrained(args.t5_model))
        setattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return args

    @staticmethod
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    @classmethod
    def _block_tri(cls, c, p):
        tri_c = cls._get_ngrams(3, c.split())
        for s in p:
            tri_s = cls._get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    @staticmethod
    def _get_abs_model_inputs(data, args, train=False):
        device = args.device
        tokenizer = args.tokenizer
        source_encoding = tokenizer.batch_encode_plus(
            data['article'],
            max_length=args.abs_max_input_len,
            pad_to_max_length=True,
            return_tensors='pt',
        )
        results = {
            'input_ids': source_encoding.input_ids.to(device),
            'attention_mask': source_encoding.attention_mask.to(device),
        }
        if train:
            results['lm_labels'] = torch.tensor(
                [
                    [
                        (label if label != tokenizer.pad_token_id else -100)
                        for label in labels_example
                    ] for labels_example in tokenizer.batch_encode_plus(
                        data['highlights'],
                        max_length= args.abs_max_output_len,
                        pad_to_max_length=True,
                        return_tensors='pt',
                    ).input_ids
                ],
                device=device,
            )
        return results

    @staticmethod
    def _get_abs_model(args, train=False):
        if train:
            pattern = 'epoch_*_batch_*_model.pt'
        else:
            pattern = f'epoch_{args.abs_epochs}_model.pt'
        files = glob.glob(os.path.join(args.abs_model_path, pattern))
        if files:
            steps = []
            for fh in files:
                split = os.path.split(fh)[1].rsplit('_', 5)
                if train:
                    # so that the training doesn't skip training the rest
                    # of the epoch
                    epoch = int(split[-4]) - 1
                    batch = int(split[-2])
                else:
                    epoch = int(split[-2])
                    batch = 0
                steps.append((epoch, batch))
            if train:
                max_batch = 0
                max_epoch = 0
                for iii, (epoch, batch) in enumerate(steps):
                    if epoch > max_epoch:
                        max_epoch = epoch
                        max_batch = batch
                        idx = iii
                    elif epoch == max_epoch and batch > max_batch:
                        max_batch = batch
                        idx = iii
            else:
                idx = 0
            model = torch.load(files[idx])
            epoch, batch = steps[idx]
        else:
            if not train:
                raise Exception(
                    f'Could not find model for epoch {args.abs_epochs}'
                )
            model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
            epoch, batch = 0, 0
        model = model.to(args.device)
        return model, epoch, batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t5_model', default='t5-large')
    parser.add_argument(
        '-ext_model_path',
        default=os.path.join(
            DIR, 'PreSumm', 'models', 'cnndm_ext', 'cnndm',
            'model_step_45000.pt'
        ),
        help='Path to the extractive summary model.',
    )
    parser.add_argument(
        '-abs_model_path', default=None,
    )
    parser.add_argument(
        '-do_extraction', type=str2bool, nargs='?', const=True, default=True,
    )
    parser.add_argument(
        '-train_abs', type=str2bool, nargs='?', const=True, default=True,
    )
    parser.add_argument(
        '-gen_summaries', type=str2bool, nargs='?', const=True, default=True,
    )
    parser.add_argument(
        '-do_evaluation', type=str2bool, nargs='?', const=True, default=True,
    )
    parser.add_argument(
        '-output_dir', default=os.path.join(DIR, 'hybrid_outputs')
    )
    parser.add_argument('-abs_max_input_len', default=512, type=int)
    parser.add_argument('-abs_min_output_len', default=40, type=int)
    parser.add_argument('-abs_max_output_len', default=150, type=int)
    parser.add_argument('-abs_epochs', default=10000, type=int)
    parser.add_argument('-abs_learning_rate', default=1e-4, type=float)
    parser.add_argument('-abs_batch_size', default=500, type=int)
    parser.add_argument('-abs_num_beams', default=2, type=int)
    parser.add_argument('-abs_rep_penalty', default=2.5, type=float)
    parser.add_argument('-abs_length_penalty', default=1.0, type=float)
    parser.add_argument(
        '-abs_early_stopping',
        default=True,
        type=str2bool,
        nargs='?',
        const=True,
    )


    ### BEGIN: parameters from PreSumm ###
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)



    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)


    ### END: parameters from PreSumm ###


    args = parser.parse_args()
    HybridModel.main(args)
