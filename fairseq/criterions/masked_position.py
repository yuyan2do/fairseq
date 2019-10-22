# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('masked_position')
class MaskedPositionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        token_sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if token_sample_size == 0:
            masked_tokens = None

        # compute Masked Position loss
        masked_positions = sample['target_position'].ne(self.padding_idx)
        position_sample_size = masked_positions.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if position_sample_size == 0:
            masked_positions = None

        positions = sample['net_input']['src_positions']
        logits, extra = model(**sample['net_input'], masked_tokens=masked_tokens, masked_positions=masked_positions, positions=positions)
        position_logits = extra['position_logits']

        targets = model.get_targets(sample, [logits])
        if token_sample_size != 0:
            targets = targets[masked_tokens]

        token_loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        targets_position= sample['target_position']
        if position_sample_size != 0:
            targets_position = targets_position[masked_positions]

        position_log_predict = \
            F.log_softmax(
                position_logits.view(-1, position_logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            )

        if position_sample_size != 0:
            position_predict_correct_num = (torch.argmax(position_log_predict, dim=-1) == targets_position.view(-1)).sum().item()
        else:
            position_predict_correct_num = 0

        position_accuracy = float(position_predict_correct_num) / max(1, position_sample_size)

        position_loss = F.nll_loss(
            position_log_predict,
            targets_position.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        sample_size = token_sample_size + position_sample_size
        loss = token_loss + position_loss

        logging_output = {
            'token_loss': utils.item(token_loss.data) if reduce else token_loss.data,
            'position_loss': utils.item(position_loss.data) if reduce else position_loss.data,
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'token_sample_size': token_sample_size,
            'position_sample_size': position_sample_size,
            'position_predict_correct_num': position_predict_correct_num,
            'position_accuracy': position_accuracy,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        token_loss = sum(log.get('token_loss', 0) for log in logging_outputs)
        position_loss = sum(log.get('position_loss', 0) for log in logging_outputs)
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        token_sample_size = sum(log.get('token_sample_size', 0) for log in logging_outputs)
        position_sample_size = sum(log.get('position_sample_size', 0) for log in logging_outputs)
        position_predict_correct_num = sum(log.get('position_predict_correct_num', 0) for log in logging_outputs)

        agg_output = {
            'token_loss': token_loss / token_sample_size / math.log(2),
            'position_loss': position_loss / max(1, position_sample_size) / math.log(2),
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'token_sample_size': token_sample_size,
            'position_sample_size': position_sample_size,
            'position_accuracy': float(position_predict_correct_num) / max(1, position_sample_size),
        }
        return agg_output
