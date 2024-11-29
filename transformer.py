# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
# from collections import _VT_co
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List
from einops import rearrange
import numbers     


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)

        self.qkv_layout = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv_layout = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.incor = ResnetBlock_Spade(256, layout_dim=256, dilation=1, use_spectral_norm=False)
        
    def forward(self, x, layout):
        # import pdb;pdb.set_trace()
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1) 
        # print(q.shape)  
        
        # import pdb;pdb.set_trace()
        ### for layout ###
        b_l, c_l, h_l, w_l = layout.shape

        qkv_lay = self.qkv_dwconv_layout(self.qkv_layout(layout))
        q_lay,k_lay,v_lay = qkv_lay.chunk(3, dim=1)

        q = self.incor(q, q_lay)
        k = self.incor(k, k_lay)
        v = self.incor(v, v_lay)
        ### for layout ###
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # import pdb;pdb.set_trace()
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, layout):
        x = x + self.attn(self.norm1(x), layout)
        x = x + self.ffn(self.norm2(x))

        return x

class ResnetBlock_Spade(nn.Module):
    def __init__(self, dim, layout_dim, dilation, use_spectral_norm=True):
        super(ResnetBlock_Spade, self).__init__()
        self.conv_block = nn.Sequential(
            SPADE (dim, layout_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),

            # SPADE(256, layout_dim),
            # nn.ReLU(True),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            # RN_L(feature_channels=dim, threshold=threshold),
        )

    def forward(self, x, layout):
        # out = x + self.conv_block(x)
        out = x
        for i in range(len(self.conv_block)):
            sub_block = self.conv_block[i]
            if i == 0 or i == 4:
                out = sub_block(out, layout)
            else:
                out = sub_block(out)
    
        out_final = out + x
        # skimage.io.imsave('block.png', out[0].detach().permute(1,2,0).cpu().numpy()[:,:,0])

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out_final

class SPADE(nn.Module):

    def __init__(self, norm_nc, label_nc):

        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        #actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class LGCM(nn.Module):
    def __init__(self, LayerNorm_type = 'WithBias'):
        super(LGCM, self).__init__()

        self.TE = nn.Sequential(*[TransformerBlock(dim=int(256), num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=LayerNorm_type) for i in range(8)])
        self.feat_mapping_layer = nn.Conv2d(512,256,kernel_size=1)

    def forward(self, x, middle_feat):
        hs = x
        inter_out = []

        layout = self.feat_mapping_layer(torch.cat(middle_feat[-1], 1))

        for i in range(len(self.TE)):
            sub_block = self.TE[i]
            hs = sub_block(hs, layout)
            inter_out.append(hs)
        return hs, inter_out

def build_transformer(args):
    return LGCM()


# class DecoderEmbeddings(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(
#             vocab_size, hidden_dim, padding_idx=pad_token_id)
#         self.position_embeddings = nn.Embedding(
#             max_position_embeddings, hidden_dim
#         )

#         self.LayerNorm = torch.nn.LayerNorm(
#             hidden_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         input_shape = x.size()
#         seq_length = input_shape[1]
#         device = x.device

#         position_ids = torch.arange(
#             seq_length, dtype=torch.long, device=device)
#         position_ids = position_ids.unsqueeze(0).expand(input_shape)

#         input_embeds = self.word_embeddings(x)
#         position_embeds = self.position_embeddings(position_ids)

#         embeddings = input_embeds + position_embeds
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)

#         return embeddings

# class Transformer(nn.Module):

#     def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
#                  num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False,
#                  return_intermediate_dec=False):
#         super().__init__()
#         encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
#                                                 dropout, activation, normalize_before)
#         encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
#         self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

#         # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
#         #                                         dropout, activation, normalize_before)
#         # decoder_norm = nn.LayerNorm(d_model)
#         # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
#         #                                 return_intermediate=return_intermediate_dec)
                                        
#         self._reset_parameters()
#         self.d_model = d_model
#         self.nhead = nhead

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, src, mask, pos_embed):
#         # flatten NxCxHxW to HWxNxC
#         bs, c, h, w = src.shape
#         src = src.flatten(2).permute(2, 0, 1)
#         pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
#         mask = mask.flatten(1)
#         # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
#         memory, inter_out = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
#         # import pdb;pdb.set_trace()
#         # tgt = torch.zeros_like(query_embed)
#         # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
#         #                 pos=pos_embed, query_pos=query_embed)
#         # import pdb;pdb.set_trace()
#         # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
#         return memory.permute(1, 2, 0).view(bs, c, h, w), inter_out


# def generate_square_subsequent_mask(sz):
#     r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
#         Unmasked positions are filled with float(0.0).
#     """
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float(
#         '-inf')).masked_fill(mask == 1, float(0.0))
#     return mask

# class TransformerEncoder(nn.Module):

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src,
#                 mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         output = src

#         inter_out = []

#         for layer in self.layers:
#             output = layer(output, src_mask=mask,
#                            src_key_padding_mask=src_key_padding_mask, pos=pos)
#             inter_out.append(output)
#         # import pdb;pdb.set_trace()
#         if self.norm is not None:
#             output = self.norm(output)

#         return output, inter_out


# class TransformerDecoder(nn.Module):

#     def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate

#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         output = tgt

#         intermediate = []

#         for layer in self.layers:
#             output = layer(output, memory, tgt_mask=tgt_mask,
#                            memory_mask=memory_mask,
#                            tgt_key_padding_mask=tgt_key_padding_mask,
#                            memory_key_padding_mask=memory_key_padding_mask,
#                            pos=pos, query_pos=query_pos)
#             if self.return_intermediate:
#                 intermediate.append(self.norm(output))

#         if self.norm is not None:
#             output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)

#         if self.return_intermediate:
#             return torch.stack(intermediate)

#         return output.unsqueeze(0)


# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self,
#                      src,
#                      src_mask: Optional[Tensor] = None,
#                      src_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None):
#         q = k = self.with_pos_embed(src, pos)
#         src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         # import pdb;pdb.set_trace()
#         return src

#     def forward_pre(self, src,
#                     src_mask: Optional[Tensor] = None,
#                     src_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None):
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#         src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         src = src + self.dropout2(src2)
#         return src

#     def forward(self, src,
#                 src_mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
#         return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# class TransformerDecoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self, tgt, memory,
#                      tgt_mask: Optional[Tensor] = None,
#                      memory_mask: Optional[Tensor] = None,
#                      tgt_key_padding_mask: Optional[Tensor] = None,
#                      memory_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None,
#                      query_pos: Optional[Tensor] = None):
#         q = k = self.with_pos_embed(tgt, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt

#     def forward_pre(self, tgt, memory,
#                     tgt_mask: Optional[Tensor] = None,
#                     memory_mask: Optional[Tensor] = None,
#                     tgt_key_padding_mask: Optional[Tensor] = None,
#                     memory_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None,
#                     query_pos: Optional[Tensor] = None):
#         tgt2 = self.norm1(tgt)
#         q = k = self.with_pos_embed(tgt2, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.norm3(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt

#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
#                                     tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
#         return self.forward_post(tgt, memory, tgt_mask, memory_mask,
#                                  tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# def build_transformer(args):
        
#     return Transformer(
#         d_model=args.hidden_dim,
#         dropout=args.dropout,
#         nhead=args.nheads,
#         dim_feedforward=args.dim_feedforward,
#         num_encoder_layers=args.enc_layers,
#         num_decoder_layers=args.dec_layers,
#         normalize_before=args.pre_norm,
#         return_intermediate_dec=True,
#     )

# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
