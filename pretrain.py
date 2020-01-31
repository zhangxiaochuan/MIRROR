# -*- coding: utf-8 -*- 
# @Time : 2019-10-29 09:53 
# @Author : Xiaochuan Zhang

"""
Pretrain Transformer
"""
from networks import Transformer
import utils
from loss import cal_translator_performance
from tqdm import trange
import os
import json
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random
from data_manager import DataLoader, Dataset


def train(model, data_iterator, optimizer, scheduler, params):
    model.train()
    scheduler.step()

    precision_avg = utils.RunningAverage()
    loss_avg = utils.RunningAverage()

    t = trange(params.train_steps, desc="Train: ")
    for _ in t:
        # fetch the next training batch
        sources, source_pos, targets, target_pos = next(data_iterator)
        preds = model(sources, source_pos, targets, target_pos)

        gold = targets[:, 1:]
        loss, precision = cal_translator_performance(preds, gold)
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        loss_avg.update(loss.item())
        precision_avg.update(precision)
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()), precision='{:05.3f}'.format(precision_avg()))
    return loss_avg(), precision_avg()


def evaluate(model, data_iterator, params):
    model.eval()

    precision_avg = utils.RunningAverage()
    loss_avg = utils.RunningAverage()

    t = trange(params.val_steps, desc="Evaluate: ")
    for _ in t:
        # fetch the next evaluation batch
        sources, source_pos, targets, target_pos = next(data_iterator)
        preds = model(sources, source_pos, targets, target_pos)

        gold = targets[:, 1:]
        loss, precision = cal_translator_performance(preds, gold)

        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()

        loss_avg.update(loss.item())
        precision_avg.update(precision)

        t.set_postfix(loss='{:05.3f}'.format(loss_avg()), precision='{:05.3f}'.format(precision_avg()))
    return loss_avg(), precision_avg()


def train_and_evaluate():

    # Preparation
    file_path = os.path.realpath(__file__)
    base_dir = os.path.dirname(file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    params = utils.Params(os.path.join(base_dir, "transformer_params.json"))

    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()

    # Set the random seed for reproducible experiments
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(params.seed)  # set random seed for all GPUs

    data, n_source_vocab, n_target_vocab = Dataset().load()
    data_loader = DataLoader(data, params.batch_size, require_negative_samples=False, seed=params.seed)
    transformer_model_dir = os.path.join(base_dir, './pretrained_models', 'transformer')
    print("max len: ", data_loader.max_len)
    transformer_config = {'n_source_vocab': n_source_vocab,
                          'n_target_vocab': n_target_vocab,
                          'max_len': data_loader.max_len,
                          'd_word_vec': 256,
                          'd_inner': 2048,
                          'n_layers': 6,
                          'n_head': 8,
                          'dropout': 0.1}

    transformer = Transformer(n_source_vocab=transformer_config['n_source_vocab'],
                              n_target_vocab=transformer_config['n_target_vocab'],
                              max_len=transformer_config['max_len'],
                              d_word_vec=transformer_config['d_word_vec'],
                              d_inner=transformer_config['d_inner'],
                              n_layers=transformer_config['n_layers'],
                              n_head=transformer_config['n_head'],
                              dropout=transformer_config['dropout'])

    transformer.to(params.device)
    if params.n_gpu > 1 and params.multi_gpu:
        transformer = torch.nn.DataParallel(transformer)

    # Prepare optimizer
    optimizer = Adam(filter(lambda x: x.requires_grad, transformer.parameters()), lr=params.learning_rate,
                     betas=(0.9, 0.98), eps=1e-09)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    history = {"train_loss": [], "val_loss": [], "train_precision": [], "val_precision": []}
    """Train the model and evaluate every epoch."""
    for epoch in range(1, params.epoch_num + 1):
        print("Epoch: " + str(epoch) + "/" + str(params.epoch_num))
        # Compute number of batches in one epoch
        train_size, val_size = data_loader.get_train_and_val_size()
        params.train_steps = train_size // params.batch_size
        params.val_steps = val_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator("train", shuffle=True)
        val_data_iterator = data_loader.data_iterator("val", shuffle=False)

        train_loss, train_precision = train(transformer, train_data_iterator, optimizer, scheduler, params)
        val_loss, val_precision = evaluate(transformer, val_data_iterator, params)
        history["train_loss"].append(train_loss)
        history["train_precision"].append(train_precision)
        history["val_loss"].append(val_loss)
        history["val_precision"].append(val_precision)

        # Save weights of the network
        model_to_save = transformer.module if hasattr(transformer, 'module') else transformer  # Only save the model it-self

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              transformer_config,
                              is_best=(val_loss == min(history["val_loss"])),
                              checkpoint=transformer_model_dir)
    with open(os.path.join(transformer_model_dir, 'history.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    train_and_evaluate()
