# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch

from fairseq.data import data_utils, Dictionary

from . import BaseWrapperDataset, LRUCacheDataset


class MaskPositionsDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False, mask_position=True, return_positions=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False, mask_position=True, return_positions=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
        mask_position: bool = False,
        mask_position_prob: float = 0.05,
        geo_distribution_prob: float = 0.125,
        return_positions: bool = False,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert 0.0 < mask_position_prob < 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        self.mask_position = mask_position
        self.mask_position_prob = mask_position_prob
        self.return_positions = return_positions
        self.geo = torch.distributions.geometric.Geometric(geo_distribution_prob)

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[:self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            assert self.mask_idx not in item, \
                'Dataset contains mask_idx (={}), this is not expected!'.format(
                    self.mask_idx,
                )

            #assert not self.mask_whole_words, 'mask whole words not support in this dataset'

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            # decide elements to mask
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz + np.random.rand()
            )
            mask[np.random.choice(sz, num_mask, replace=False)] = True

            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                if self.mask_whole_words is not None:
                    mask = np.repeat(mask, word_lens)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            if self.mask_position:
                # mask 3 - 6 token
                masked_positions = np.arange(1, 1 + len(item)) + self.pad_idx
                new_item = np.full(len(item), self.pad_idx)


                # def softmax(x):
                #    return np.exp((x - max(x))) / sum(np.exp((x - max(x))))

                num_position_mask = int(
                    # add a random number for probabilistic rounding
                    self.mask_position_prob * sz + np.random.rand()
                )

                position_span_list = [0]
                while sum(position_span_list) < num_position_mask:
                    position_span = np.clip(int(self.geo.sample().item()) + 1, 1, 12)
                    position_span_list.append(position_span)

                if np.random.rand() <= 0.5:
                    position_span_list.pop()

                position_span_list = np.sort(position_span_list)[::-1]

                for position_span in position_span_list:
                    retry = 0
                    while position_span > 0 and retry < 10:
                        # front position has higher probability been mask. token 1 has 5 times probability then last word token.
                        # start_index_probability = np.cumsum(-1*np.ones(sz - 2 - (num_position_mask - 1))) / (sz/2)
                        # start_index_probability = softmax(start_index_probability)
                        # start_index = np.random.choice(sz, 1, p=np.concatenate(([0], start_index_probability, [0] * num_position_mask)))[0]
                        if sz - position_span - 1 < 3:
                            break

                        start_index = np.random.choice(sz - position_span - 1, 1)[0] + 1
                        end_index = start_index+position_span

                        if self.mask_whole_words is not None:
                            start_index = word_begins_idx[start_index].item()
                            end_index = word_begins_idx[end_index].item()

                        if mask[start_index:end_index].any() and (masked_positions[start_index:end_index] == 0).any():
                            retry = retry + 1
                            continue

                        new_item[start_index:end_index] = masked_positions[start_index:end_index] 

                        #unmask_position = np.random.rand(num_position_mask) < 0.15
                        #masked_positions[start_index:end_index][~unmask_position] = [0] * (num_position_mask - unmask_position.sum())
                        masked_positions[start_index:end_index] = [0] * (end_index - start_index)
                        break

                if self.return_positions:
                    return torch.from_numpy(new_item)
                else:
                    return torch.from_numpy(masked_positions)


            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )

            return torch.from_numpy(new_item)
