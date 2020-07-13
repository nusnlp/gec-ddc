# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
from rl import departureLossRL
import torch


@register_criterion('label_smooth_ce_rl')
class LabelSmoothedCrossEntropyReinforceCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

        # self.kakao = None #Kakao_model()
        # self.edin = None #Edin_model()

        self.loss_rl = departureLossRL(device=None, word_alphabet=None, vocab_size=None, args=args)  # no .cuda()??

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        rl_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        rl_loss = rl_loss*1000
        loss = loss + rl_loss
        #print('loss: ', loss)
        #print('rl_loss: ', rl_loss)
        # loss = rl_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        sel = torch.argmax(lprobs, dim=2, keepdim=False)
        lprobs_size = lprobs.size()
        pb = torch.gather(lprobs, 2, sel.view((lprobs_size[0], lprobs_size[1], 1))).squeeze(2)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1)

        out = sel  # (7, 20)
        rl_loss, _, _ = self.loss_rl(sel, pb=pb, out=out, mask_id=None, stc_length_out=None, sample_input=sample, ignore_index=self.padding_idx,)
        return rl_loss, rl_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size = 1 if sample_size==0 else sample_size
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
