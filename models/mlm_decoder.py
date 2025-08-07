# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True,  # 初始化时，normalize_before 默认为 false
                 return_intermediate_dec=False):
        super().__init__()
        assert normalize_before == True
        print("\nmlm_head normalize_before: ", normalize_before)

        # Please note that all the "decoders" mentioned here are renamed from the "condition encoders" in the HiVG work.
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.adapt_layer = [i for i in range(num_encoder_layers)]

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, memory, mask=None, memory_mask=None, memory_pos=None, pos_embed=None,
                adapt_layer=None, query_embed=None):
        # # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # mask = mask.flatten(1)

        if adapt_layer is None:
            adapt_layer = self.adapt_layer

        memory = self.decoder(src, memory,
                              adapt_layer=adapt_layer,
                              src_key_padding_mask=mask,
                              memory_key_padding_mask=memory_mask,
                              pos=pos_embed,
                              memory_pos=memory_pos)

        return memory


class TransformerDecoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, memory, adapt_layer,
                mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None):
        output = src

        layer_num = 0
        for layer in self.layers:
            output = layer(output, memory, layer=layer_num, adapt_layer=adapt_layer, src_mask=mask, memory_mask=memory_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           memory_pos=memory_pos)
            layer_num += 1

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # 新增
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post_ori(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_post(self,
                     src,
                     memory,
                     layer, adapt_layer,
                     src_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     memory_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 新增
        if layer in adapt_layer:
            tgt2 = self.cross_attn(query=self.with_pos_embed(src, pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
            src = src + self.dropout2(tgt2)
            src = self.norm2(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        return src

    def forward_pre_ori(self, src,
                        src_mask: Optional[Tensor] = None,
                        src_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward_pre(self,
                    src,
                    memory,
                    layer, adapt_layer,
                    src_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    memory_pos: Optional[Tensor] = None):

        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # 新增
        if layer in adapt_layer:
            src2 = self.norm2(src)
            tgt2 = self.cross_attn(query=self.with_pos_embed(src2, pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
            src = src + self.dropout2(tgt2)

        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        return src

    def forward(self, src, memory, layer, adapt_layer,
                src_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, memory, layer, adapt_layer, src_mask, memory_mask,
                                    src_key_padding_mask, memory_key_padding_mask, pos, memory_pos)

        return self.forward_post(src, memory, layer, adapt_layer, src_mask, memory_mask,
                                 src_key_padding_mask, memory_key_padding_mask, pos, memory_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(hidden_dim, dropout, nheads, dim_feedforward, num_encoder_layers,
                      num_decoder_layers=0, normalize_before=True):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        normalize_before=normalize_before,  # default True
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



