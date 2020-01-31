# -*- coding: utf-8 -*- 
# @Time : 2019-10-21 20:38 
# @Author : Xiaochuan Zhang

import torch
import torch.nn.functional as F
import torch.nn as nn


def cal_translator_performance(preds, target, smoothing=False):
    """Apply label smoothing if needed"""

    loss = cal_translator_loss(preds, target, smoothing)

    preds = preds.max(1)[1]
    gold = target.contiguous().view(-1)
    non_pad_mask = gold.ne(0)
    n_correct = preds.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    precision = n_correct / non_pad_mask.sum().item()
    return loss, precision


def cal_translator_loss(pred, target, smoothing):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    target = target.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = target.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, target, ignore_index=0, reduction='sum')

    return loss


def cal_triplet_margin_loss(anchor, positive, negative, margin):
    """
    Calculate triplet margin loss
    :param anchor: the embedding of anchors
    :param positive: the embedding of positive samples
    :param negative: the embedding of negative samples
    :param margin: margin
    :return: loss
    """
    return nn.TripletMarginLoss(margin=margin)(anchor, positive, negative)
