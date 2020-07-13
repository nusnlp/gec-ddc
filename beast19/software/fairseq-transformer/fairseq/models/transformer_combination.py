# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel, FairseqDualEncoderModel,
    register_model, register_model_architecture,
)


@register_model('transformer_combination')
class TransformerModel(FairseqDualEncoderModel):
    def __init__(self, auxencoder, encoder, decoder):
        super().__init__(auxencoder, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--context-token-dropout', default=0.0, type=float, metavar='D',
                            help='dropout probability to drop complete words of context')
        parser.add_argument('--source-token-dropout', default=0.0, type=float, metavar='D',
                            help='dropout probability to drop complete words')
        parser.add_argument('--target-token-dropout', default=0.0, type=float, metavar='D',
                            help='dropout probability to drop complete words')
        parser.add_argument('--auxencoder-embed-dim', type=int, metavar='N',
                            help='auxencoder embedding dimension')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--auxencoder-ffn-embed-dim', type=int, metavar='N',
                            help='auxencoder embedding dimension for FFN')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--auxencoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained auxencoder embedding')
        parser.add_argument('--encoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--auxencoder-layers', type=int, metavar='N',
                            help='num auxencoder layers')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        ctx_dict, src_dict, tgt_dict = task.context_dictionary, task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, embed_dict=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()

            embed = Embedding(num_embeddings, embed_dim, padding_idx)
            if embed_dict:
                embed = utils.load_embedding(embed_dict, dictionary, embed)

            return embed

        auxencoder_embed_dict = None
        if args.auxencoder_embed_path:
            auxencoder_embed_dict = utils.parse_embedding(args.auxencoder_embed_path)
            utils.print_embed_overlap(auxencoder_embed_dict, task.context_dictionary)

        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)


        if args.share_all_embeddings:
            if ctx_dict != src_dict or src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.auxencoder_embed_dim != args.encoder_embed_dim or args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --auxencoder-embed-dim and --decoder-embed-dim')
            if args.auxencoder_embed_path != args.encoder_embed_path or args.encoder_embed_path != args.decoder_embed_path:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-path to match --auxencoder-embed-path and --decoder-embed-path')
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, embed_dict=encoder_embed_dict)
            auxencoder_embed_tokens = encoder_embed_tokens
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            auxencoder_embed_tokens = build_embedding(ctx_dict, args.auxencoder_embed_dim, embed_dict=auxencoder_embed_dict)
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, embed_dict=encoder_embed_dict)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, embed_dict=decoder_embed_dict)

        auxencoder = TransformerEncoder(args, ctx_dict, auxencoder_embed_tokens, auxenc=True)
        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(auxencoder, encoder, decoder)


class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=True, auxenc=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        if auxenc:
            self.token_dropout = args.context_token_dropout
        else:
            self.token_dropout = args.source_token_dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        if auxenc:
            self.layers.extend([
                TransformerEncoderLayer(args, auxenc=True)
                for i in range(args.auxencoder_layers)
            ])
        else:
            self.layers.extend([
                TransformerEncoderLayer(args)
                for i in range(args.encoder_layers)
            ])

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        # drops certain words with probability token_dropout
        x = F.dropout2d(x, p=self.token_dropout, training=self.training)

        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            if 'encoder.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.token_dropout = args.target_token_dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for i in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, prev_output_tokens, auxencoder_out, encoder_out, incremental_state=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = F.dropout2d(x, p=self.token_dropout, training=self.training)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                auxencoder_out['encoder_out'],
                auxencoder_out['encoder_padding_mask'],
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        return x, attn

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def reorder_auxencoder_out(self, auxencoder_out_dict, new_order):
        if auxencoder_out_dict['encoder_padding_mask'] is not None:
            auxencoder_out_dict['encoder_padding_mask'] = \
                auxencoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return auxencoder_out_dict

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, args, auxenc=False):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        if auxenc:
	        self.fc1 = Linear(self.embed_dim, args.auxencoder_ffn_embed_dim)
	        self.fc2 = Linear(args.auxencoder_ffn_embed_dim, self.embed_dim)
        else:
        	self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        	self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        self.auxencoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, auxencoder_out, auxencoder_padding_mask, encoder_out, encoder_padding_mask, incremental_state):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)

        x_enc = x
        x_enc, attn_enc = self.encoder_attn(
            query=x_enc,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
        )
        x_enc = F.dropout(x_enc, p=self.dropout, training=self.training)

        x_auxenc = x
        x_auxenc, attn_auxenc = self.auxencoder_attn(
            query=x_auxenc,
            key=auxencoder_out,
            value=auxencoder_out,
            key_padding_mask=auxencoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
        )
        x_auxenc = F.dropout(x_auxenc, p=self.dropout, training=self.training)

        x = residual + x_enc + x_auxenc
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn_enc

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, init_size=num_embeddings)
    return m


@register_model_architecture('transformer_combination', 'transformer_combination')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.auxencoder_embed_dim = getattr(args, 'auxencoder_embed_dim', 1024)
    args.auxencoder_ffn_embed_dim = getattr(args, 'auxencoder_ffn_embed_dim', 2048)
    args.auxencoder_layers = getattr(args, 'auxencoder_layers', 1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)


# @register_model_architecture('transformer', 'transformer_iwslt_de_en')
# def transformer_iwslt_de_en(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_layers = getattr(args, 'encoder_layers', 3)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
#     args.decoder_layers = getattr(args, 'decoder_layers', 3)
#     base_architecture(args)


# @register_model_architecture('transformer', 'transformer_wmt_en_de')
# def transformer_wmt_en_de(args):
#     base_architecture(args)


# # parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
# @register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
# def transformer_vaswani_wmt_en_de_big(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     args.dropout = getattr(args, 'dropout', 0.3)
#     base_architecture(args)


# @register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
# def transformer_vaswani_wmt_en_fr_big(args):
#     args.dropout = getattr(args, 'dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# @register_model_architecture('transformer', 'transformer_wmt_en_de_big')
# def transformer_wmt_en_de_big(args):
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


# # default parameters used in tensor2tensor implementation
# @register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
# def transformer_wmt_en_de_big_t2t(args):
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
#     args.encoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
#     args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
#     args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
#     transformer_vaswani_wmt_en_de_big(args)
