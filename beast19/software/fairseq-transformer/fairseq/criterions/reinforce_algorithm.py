# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F

from fairseq import utils
# from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from rl import departureLossRL
import torch
# from Kakao import Kakao_model
# from Edin import Edin_model
# from Toho import Toho_model

@register_criterion('reinforce_criterion')
class ReinforceCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        self.loss_rl = departureLossRL(device='cpu', word_alphabet=None, vocab_size=None)   # no .cuda()??
        self.kakao = None #Kakao_model()
        self.edin = None #Edin_model()
        self.toho = None #Toho_model()
        self.departure_reference_models = [self.kakao, self.edin, self.toho]

    def departure_reference_models(self, sample):
        return None  # TODO: hanwj

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        sel = torch.argmax(lprobs, dim=2, keepdim=False)
        lprobs_size = lprobs.size()
        pb = torch.gather(lprobs, 2, sel.view((lprobs_size[0], lprobs_size[1], 1))).squeeze(2)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        golden_out = sel  # (7, 20)
        # src_stc_length_out = sample['net_input']['src_lengths'] # (7,)

        departure_out_list = self.departure_reference_models(sample)
        # departure_out_list = golden_out#None
        rl_loss, _, _ = self.loss_rl(sel, pb=pb, golden_out=golden_out, mask_id=None, stc_length_out=None, departure_out_list=departure_out_list, ignore_index=self.padding_idx,)
        return nll_loss+rl_loss, nll_loss+rl_loss

    # @staticmethod
    # def reduce_metrics(logging_outputs) -> None:
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
    #     ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
    #     sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

    #     metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
    #     if sample_size != ntokens:
    #         metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
    #         metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
    #     else:
    #         metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
