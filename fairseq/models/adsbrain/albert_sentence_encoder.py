# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
)

from fairseq import utils
from .expend_multihead_attention import ExpendMultiheadAttention

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, ExpendMultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)


class AlbertSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        word_dim: int = 128,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dim_multiplier: int = 1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.word_dim = word_dim
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.num_encoder_layers = num_encoder_layers

        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.word_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.word_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.word_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.word_dim != self.embedding_dim:
            self.fc = nn.Linear(self.word_dim, self.embedding_dim)
        else:
            self.fc = None

        self.layer = AlbertSentenceEncoderLayer(
                          embedding_dim=self.embedding_dim,
                          ffn_embedding_dim=ffn_embedding_dim,
                          num_attention_heads=num_attention_heads,
                          dim_multiplier=dim_multiplier,
                          dropout=self.dropout,
                          attention_dropout=attention_dropout,
                          activation_dropout=activation_dropout,
                          activation_fn=activation_fn,
                          add_bias_kv=add_bias_kv,
                          add_zero_attn=add_zero_attn,
                          export=export,
                          num_encoder_layers = self.num_encoder_layers
                      )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.word_dim, export=export)
        else:
            self.emb_layer_norm = None

        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)
            # follow Generating Long Sequences with Sparse Transformers
            # to keep the ratio of embedding scale to residual block scale invariant across layer
            self.layer.self_attn.out_proj.weight.data.normal_(mean=0.0, std=0.02 / torch.sqrt(torch.tensor(2 * self.num_encoder_layers, dtype=torch.float)))
            self.layer.fc2.weight.data.normal_(mean=0.0, std=0.02 / torch.sqrt(torch.tensor(2 * self.num_encoder_layers, dtype=torch.float)))

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        if n_trans_layers_to_freeze > 0:
            freeze_module_params(self.layer)

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        execute_times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if execute_times is None:
            execute_times = self.num_encoder_layers

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        if self.fc is not None:
            x = self.fc(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(execute_times):
            x, _ = self.layer(x, self_attn_padding_mask=padding_mask, layer=i)
            if not last_state_only:
                inner_states.append(x)

        x = self.final_layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep

class AlbertSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dim_multiplier: int = 1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
        num_encoder_layers: int = 6,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = ExpendMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            dim_multiplier=dim_multiplier,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm_list = nn.ModuleList([ LayerNorm(self.embedding_dim, export=export) for _ in range(num_encoder_layers - 1)])
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm_list = nn.ModuleList([ LayerNorm(self.embedding_dim, export=export) for _ in range(num_encoder_layers)])

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        layer: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if layer > 0:
            x = self.self_attn_layer_norm_list[layer - 1](x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = residual + x

        residual = x

        x = self.final_layer_norm_list[layer](x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = residual + x
        return x, attn
