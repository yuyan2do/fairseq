# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('masked_lm_gan')
class MaskedLmGanLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.loss_lambda = 0.1

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits_mlm = model(**sample['net_input'], masked_tokens=masked_tokens)[0]

        targets_mlm = model.get_targets(sample, [logits_mlm])
        targets_dicriminant = torch.ones_like(masked_tokens, dtype=torch.long, device=masked_tokens.get_device())

        if sample_size != 0:
            targets_mlm = targets_mlm[masked_tokens]
            predict_token = torch.argmax(logits_mlm, dim=-1)
            match_mlm = (targets_mlm == predict_token)
            targets_dicriminant[masked_tokens][~match_mlm] = 0
            match_mlm_cnt = match_mlm.sum().item()

            # print('before', sample['net_input']['src_tokens'][masked_tokens])
            gan_token = sample['net_input']['src_tokens'].detach().clone()
            gan_token[masked_tokens] = predict_token
            # print('after', gan_token[masked_tokens])
        else:
            match_mlm_cnt = 0

        mlm_sample_size = sample_size
        loss_mlm = F.nll_loss(
            F.log_softmax(
                logits_mlm.view(-1, logits_mlm.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets_mlm.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )


        loss = loss_mlm
        if float(match_mlm_cnt) / mlm_sample_size > 0.3:
            logits_dicriminant = model(gan_token)[0]

            targets_dicriminant[sample['net_input']['src_tokens'].eq(self.padding_idx)] = 2
            match_dicriminant_cnt = (targets_dicriminant == torch.argmax(logits_dicriminant, dim=-1)).sum().item()

            loss_dicriminant = F.nll_loss(
                F.log_softmax(
                    logits_dicriminant.view(-1, logits_dicriminant.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets_dicriminant.view(-1),
                reduction='sum',
                ignore_index=2,
            )

            if loss_dicriminant < 0.1:
                self.loss_lambda = 7

            loss += self.loss_lambda * loss_dicriminant
        else:
            match_dicriminant_cnt = 0
            loss_dicriminant = torch.tensor(0)


        sample_size = mlm_sample_size
        # sample_size = mlm_sample_size + sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss_mlm': utils.item(loss_mlm.data) if reduce else los_mlms.data,
            'loss_dicriminant': utils.item(loss_dicriminant.data) if reduce else loss_dicriminant.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'mlm_sample_size': mlm_sample_size,
            'match_mlm_cnt': match_mlm_cnt,
            'match_dicriminant_cnt': match_dicriminant_cnt,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        loss_mlm = sum(log.get('loss_mlm', 0) for log in logging_outputs)
        loss_dicriminant = sum(log.get('loss_dicriminant', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        mlm_sample_size = sum(log.get('mlm_sample_size', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        match_mlm_cnt = sum(log.get('match_mlm_cnt', 0) for log in logging_outputs)
        match_dicriminant_cnt = sum(log.get('match_dicriminant_cnt', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'loss_mlm': loss_mlm / mlm_sample_size / math.log(2),
            'loss_dicriminant': loss_dicriminant / ntokens / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'accuracy_mlm': float(match_mlm_cnt) / sample_size if sample_size != 0 else 0,
            'accuracy_dicriminant': float(match_dicriminant_cnt) / ntokens if ntokens != 0 else 0,
        }
        return agg_output
