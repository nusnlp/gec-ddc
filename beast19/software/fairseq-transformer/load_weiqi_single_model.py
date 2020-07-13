#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import namedtuple
import numpy as np
import sys

import torch
from torch.autograd import Variable

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
import pickle

Batch = namedtuple('Batch', 'srcs tokens lengths')
Translation = namedtuple('Translation', 'src_str hypos alignments hypo_tokens')


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, src_dict, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = data.EpochBatchIterator(
        dataset=data.LanguagePairDataset(tokens, lengths, src_dict),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, model_args = utils.load_ensemble_for_inference(model_paths, task)

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        )

    # Initialize generator
    translator = SequenceGenerator(
        models, tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk,
        minlen=args.min_len,
    )

    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    def make_result(src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            alignments=[],
            hypo_tokens=[]
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.alignments.append('A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment))))
            result.hypo_tokens.append(hypo['tokens'])  # before process like remove_bpe
        return result

    def process_batch(batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        translations = translator.generate(
            Variable(tokens),
            Variable(lengths),
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]


    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    for inputs in buffered_read(args.buffer_size):
        indices = []
        results = []
        for batch, batch_indices in make_batches(inputs, args, src_dict, models[0].max_positions()):
            indices.extend(batch_indices)
            results += process_batch(batch)

        for i in np.argsort(indices):
            result = results[i]
            print(result.src_str)
            for hypo, align in zip(result.hypos, result.alignments):
                print(hypo)
                print(align)


class load_weiqi_single(torch.nn.Module): # parsers
    def __init__(self, main_args, departure_args_path):
        super(load_weiqi_single, self).__init__()
        df = open(departure_args_path, 'rb')
        args = pickle.load(df)
        df.close()

        args.buffer_size = 0
        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        print(args)

        use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task, e.g., translation
        task = tasks.setup_task(args)

        # Load ensemble
        print('| loading model(s) from {}'.format(args.path))
        model_paths = args.path.split(':')
        models, model_args = utils.load_ensemble_for_inference(model_paths, task)

        # Set dictionaries
        src_dict = task.source_dictionary  # dic should be the same !! TODO:
        tgt_dict = task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            )

        # Initialize generator
        translator = SequenceGenerator(
            models, tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
            normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
            unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk,
            minlen=args.min_len,
        )

        if use_cuda:
            translator.cuda()

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(args.replace_unk)

        #  -------main_args------------
        # Setup task, e.g., translation
        # main_task = tasks.setup_task(main_args)
        main_task = task
        # # Set dictionaries
        # main_src_dict = main_task.source_dictionary  # dic should be the same !! TODO:
        main_src_dict = src_dict


        self.models = models
        self.args = args
        self.main_args = main_args
        self.src_dict = src_dict
        self.use_cuda = use_cuda
        self.translator = translator
        self.align_dict = align_dict
        self.tgt_dict = tgt_dict

        self.main_src_dict = main_src_dict

    def main_args_to_args(self, sample):
        batch_size, lenghts_pad = sample['net_input']['src_tokens'].size()
        lengths = sample['net_input']['src_lengths']
        tokens = [
            [
            self.main_src_dict.symbols[wd]
            for wd in stc
                if not (wd == self.main_src_dict.pad_index)
            ]
            for stc in sample['net_input']['src_tokens']
        ]
        return [' '.join(i) for i in tokens]
    
    def forward(self, sample):  # have not ignore any tokens
        if self.args.buffer_size > 1:
            print('| Sentence buffer size:', self.args.buffer_size)
        tokens = sample['net_input']['src_tokens']
        lengths = sample['net_input']['src_lengths']

        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        translations = self.translator.generate(
            Variable(tokens),
            Variable(lengths),
            maxlen=int(self.args.max_len_a * tokens.size(1) + self.args.max_len_b),
        )
        outputs = [t[0]['tokens'].cpu().tolist() for i, t in enumerate(translations)]
        # for _ in range(1):
        #     inputs = self.main_args_to_args(sample)
        #     # inputs = ['I love dog .']
        #     indices = []
        #     results = []
        #     for batch, batch_indices in make_batches(inputs, self.args, self.src_dict, self.models[0].max_positions()):
        #         indices.extend(batch_indices)
        #         results += self.process_batch(batch)
        #     for i in np.argsort(indices):
        #         result = results[i]
        #         outputs.append(result.hypo_tokens[0].cpu().tolist())
        return outputs

    def process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        translations = self.translator.generate(
            Variable(tokens),
            Variable(lengths),
            maxlen=int(self.args.max_len_a * tokens.size(1) + self.args.max_len_b),
        )

        return [self.make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

    def make_result(self, src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            alignments=[],
            hypo_tokens=[]
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), self.args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=self.align_dict,
                tgt_dict=self.tgt_dict,
                remove_bpe=self.args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.alignments.append('A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment))))
            result.hypo_tokens.append(hypo['tokens'])  # before process like remove_bpe
        return result

if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)
