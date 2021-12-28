# -*- coding: utf-8 -*- 
# @Time : 2019-10-20 20:28 
# @Author : Xiaochuan Zhang

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from layers import DecoderLayer, EncoderLayer


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table"""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, n_vocab, max_len, d_word_vec, n_layers, n_head, d_inner, dropout=0.1, embedding_weight=None):
        super(Encoder, self).__init__()
        self.d_model = d_word_vec

        self.embedding = nn.Embedding(n_vocab, d_word_vec, padding_idx=0, _weight=embedding_weight)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_len + 1, d_word_vec, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList(
            [EncoderLayer(self.d_model, d_inner, n_head, dropout=dropout) for _ in range(n_layers)])

    def no_grads(self):
        for param in self.parameters():
            param.requires_grad = False

    def grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, seq, pos, return_attns=False, encoding_mask=None):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=seq, seq_q=seq)
        non_pad_mask = get_non_pad_mask(seq)

        # -- Forward
        enc_output = self.embedding(seq) + self.position_enc(pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                encoding_mask=encoding_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    def __init__(self, n_vocab, max_len, d_word_vec, n_layers, n_head, d_inner, dropout=0.1):
        super(Decoder, self).__init__()
        d_model = d_word_vec
        self.embedding = nn.Embedding(n_vocab, d_word_vec, padding_idx=0)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_len + 1, d_word_vec, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, dropout=dropout) for _ in range(n_layers)])

    def forward(self, target_seq, target_pos, source_seq, encoder_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(target_seq)

        slf_attn_mask_subseq = get_subsequent_mask(target_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=target_seq, seq_q=target_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=source_seq, seq_q=target_seq)

        # -- Forward
        dec_output = self.embedding(target_seq) + self.position_enc(target_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, encoder_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Measurer(nn.Module):
    def __init__(self, n_source_vocab, n_target_vocab, max_len, d_word_vec=512, d_inner=2048, n_layers=6,
                 n_head=8, dropout=0.1):
        """
        :param n_source_vocab: number of tokens in source language
        :param n_target_vocab: number of tokens in target language
        :param max_len: max length of basic blocks
        :param d_word_vec: dimension of word embedding
        :param d_inner: hidden size
        :param n_layers: number of encoder layers
        :param n_head: number of multi-head attentions
        :param dropout: dropout proportion
        """
        super(Measurer, self).__init__()
        self.max_len = max_len
        self.n_target_vocab = n_target_vocab
        self.d_word_vec = d_word_vec
        self.source_encoder = Encoder(n_vocab=n_source_vocab, max_len=max_len, d_word_vec=d_word_vec,
                                      d_inner=d_inner, n_layers=n_layers, n_head=n_head, dropout=dropout)
        self.target_encoder = Encoder(n_vocab=n_target_vocab, max_len=max_len, d_word_vec=d_word_vec,
                                      d_inner=d_inner, n_layers=n_layers, n_head=n_head, dropout=dropout)
        # TODO: Whether Euclidean distance is the best?
        # self.measurement = nn.CosineSimilarity(dim=1)
        self.measurement = nn.PairwiseDistance()

    def set_source_encoder(self, source_encoder):
        self.source_encoder = source_encoder
        # self.source_encoder.no_grads()

    def set_target_embedding_weight(self, weight):
        assert list(weight.shape) == [self.n_target_vocab, self.d_word_vec], \
            'Shape of weight does not match num_embeddings and embedding_dim'
        self.target_encoder.embedding.weight = Parameter(weight)

    def forward(self, source_seq, source_pos, target_seq, target_pos, negative_seq=None, negative_pos=None, negative_encoder=None):
        source_encode = self.source_encoder(source_seq, source_pos)
        target_encode = self.target_encoder(target_seq, target_pos)
        if negative_seq is None and negative_pos is None and negative_encoder is None:
            source_encode = source_encode.sum(1)
            target_encode = target_encode.sum(1)

            distance = self.measurement(source_encode, target_encode)
            return distance/(self.d_word_vec)
        else:
            target_encoder_mask = negative_encoder.eq(0).unsqueeze(1).expand([negative_encoder.size(0), self.max_len]) # 0: csource, 1: target
            source_encoder_mask = negative_encoder.eq(1).unsqueeze(1).expand([negative_encoder.size(0), self.max_len])
            target_encoding_mask = negative_encoder.eq(0).unsqueeze(1).expand([negative_encoder.size(0), self.d_word_vec])
            source_encoding_mask = negative_encoder.eq(1).unsqueeze(1).expand([negative_encoder.size(0), self.d_word_vec])

            source_negative_seq = negative_seq.mul(target_encoder_mask.long())
            source_negative_pos = negative_pos.mul(target_encoder_mask.long())
            target_negative_seq = negative_seq.mul(source_encoder_mask.long())
            target_negative_pos = negative_pos.mul(source_encoder_mask.long())

            source_negative_encode = self.source_encoder(source_negative_seq, source_negative_pos, encoding_mask=source_encoder_mask)
            target_negative_encode = self.target_encoder(target_negative_seq, target_negative_pos, encoding_mask=target_encoder_mask)
            source_encode = source_encode.sum(1)
            target_encode = target_encode.sum(1)

            anchor = source_encode.mul(source_encoding_mask.float()) + target_encode.mul(target_encoding_mask.float())
            positive = source_encode.mul(target_encoding_mask.float()) + target_encode.mul(source_encoding_mask.float())
            negative = (source_negative_encode + target_negative_encode).sum(1)

            return anchor, positive, negative


class Transformer(nn.Module):
    def __init__(self, n_source_vocab, n_target_vocab, max_len, d_word_vec=256, d_inner=2048, n_layers=6,
                 n_head=8, dropout=0.1):
        """
        :param n_source_vocab: number of tokens in source language
        :param n_target_vocab: number of tokens in target language
        :param max_len: max length of basic blocks
        :param d_word_vec: dimension of word embedding
        :param d_inner: hidden size
        :param n_layers: number of encoder layers
        :param n_head: number of multi-head attentions
        :param dropout: dropout proportion
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_vocab=n_source_vocab, max_len=max_len, d_word_vec=d_word_vec, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head, dropout=dropout)

        self.decoder = Decoder(n_vocab=n_target_vocab, max_len=max_len, d_word_vec=d_word_vec, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head, dropout=dropout)

        self.predictor = nn.Linear(d_word_vec, n_target_vocab, bias=False)
        nn.init.xavier_normal_(self.predictor.weight)

        # Share the weight matrix between target word embedding & the final logit dense layer
        self.predictor.weight = self.decoder.embedding.weight
        self.x_logit_scale = (d_word_vec ** -0.5)

    def forward(self, source_seq, source_pos, target_seq, target_pos):
        target_seq, target_pos = target_seq[:, :-1], target_pos[:, :-1]

        enc_output, *_ = self.encoder(source_seq, source_pos)
        dec_output, *_ = self.decoder(target_seq, target_pos, source_seq, enc_output)
        seq_logit = self.predictor(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
