# -*- coding: utf-8 -*- 
# @Time : 2019-11-3 11:50
# @Author : Xiaochuan Zhang

"""
Train Measuser
"""
from networks import Measurer
import utils
from loss import cal_triplet_margin_loss
from tqdm import trange
import os
import json
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import random
from data_manager import DataLoader, Dataset
from init_models import init_transformer, init_measurer


def train(model, data_iterator, optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()
    scheduler.step()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(params.train_steps, desc="Train: ")
    for _ in t:
        # fetch the next training batch
        sources, source_pos, targets, target_pos, negatives, negative_pos, negative_encoders = next(data_iterator)
        source_encodes, target_encodes, negative_encodes = model(sources, source_pos, targets, target_pos,
                                                                 negatives, negative_pos, negative_encoders)

        loss = cal_triplet_margin_loss(source_encodes, target_encodes, negative_encodes, params.margin)
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    return loss_avg()


def evaluate(model, data_iterator, params):
    model.eval()
    loss_avg = utils.RunningAverage()

    t = trange(params.val_steps, desc="Evaluate: ")
    for _ in t:
        # fetch the next evaluation batch
        sources, source_pos, targets, target_pos, negatives, negative_pos, negative_encoders = next(data_iterator)
        source_encodes, target_encodes, negative_encodes = model(sources, source_pos, targets, target_pos,
                                                                 negatives, negative_pos, negative_encoders)

        loss = cal_triplet_margin_loss(source_encodes, target_encodes, negative_encodes, params.margin)
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        loss_avg.update(loss.item())

        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    return loss_avg()


def train_and_evaluate():

    # Preparation
    file_path = os.path.realpath(__file__)
    base_dir = os.path.dirname(file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    params = utils.Params(os.path.join(base_dir, "measurer_params.json"))

    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()

    # Set the random seed for reproducible experiments
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(params.seed)  # set random seed for all GPUs

    data, _, _ = Dataset().load()
    data_loader = DataLoader(data, params.batch_size, require_negative_samples=True, seed=params.seed)

    transformer, transformer_config = init_transformer()

    measurer_model_dir = os.path.join(base_dir, './pretrained_models', 'measurer')
    print("max len: ", data_loader.max_len)

    measurer_config = transformer_config

    measurer = Measurer(n_source_vocab=measurer_config['n_source_vocab'],
                        n_target_vocab=measurer_config['n_target_vocab'],
                        max_len=measurer_config['max_len'],
                        d_word_vec=measurer_config['d_word_vec'],
                        d_inner=measurer_config['d_inner'],
                        n_layers=measurer_config['n_layers'],
                        n_head=measurer_config['n_head'],
                        dropout=measurer_config['dropout'])

    measurer.set_source_encoder(transformer.encoder)
    # measurer, measurer_config = init_measurer()
    measurer.to(params.device)
    if params.n_gpu > 1 and params.multi_gpu:
        measurer = torch.nn.DataParallel(measurer)

    # Prepare optimizer
    optimizer = Adam(filter(lambda x: x.requires_grad, measurer.parameters()), lr=params.learning_rate,
                     betas=(0.9, 0.98), eps=1e-09)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    history = {"train_loss": [], "val_loss": []}
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

        # Train for one epoch on training set
        train_loss = train(measurer, train_data_iterator, optimizer, scheduler, params)
        val_loss = evaluate(measurer, val_data_iterator, params)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save weights of the network
        model_to_save = measurer.module if hasattr(measurer, 'module') else measurer  # Only save the model it-self

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              measurer_config,
                              is_best=(val_loss == min(history["val_loss"])),
                              checkpoint=measurer_model_dir)
    with open(os.path.join(measurer_model_dir, 'history.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    train_and_evaluate()
