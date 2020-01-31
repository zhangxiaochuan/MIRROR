# -*- coding: utf-8 -*- 
# @Time : 2019-10-25 22:04 
# @Author : Xiaochuan Zhang

import os
import json
import utils
from networks import Transformer, Measurer


def init_transformer():
    file_path = os.path.realpath(__file__)
    base_dir = os.path.dirname(file_path)
    model_dir = os.path.join(base_dir, 'pretrained_models', 'transformer')

    if os.path.exists(os.path.join(model_dir, 'best.pth.tar')):
        with open(os.path.join(model_dir, 'best_config.json')) as f:
            model_config = json.load(f)
            model = Transformer(n_source_vocab=model_config['n_source_vocab'],
                                n_target_vocab=model_config['n_target_vocab'],
                                max_len=model_config['max_len'],
                                d_word_vec=model_config['d_word_vec'],
                                d_inner=model_config['d_inner'],
                                n_layers=model_config['n_layers'],
                                n_head=model_config['n_head'],
                                dropout=model_config['dropout'])
        print("loading: ", os.path.join(model_dir, 'best.pth.tar'))
        utils.load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model)
    elif os.path.exists(os.path.join(model_dir, 'last.pth.tar')):
        with open(os.path.join(model_dir, 'last_config.json')) as f:
            model_config = json.load(f)
            model = Transformer(n_source_vocab=model_config['n_source_vocab'],
                                n_target_vocab=model_config['n_target_vocab'],
                                max_len=model_config['max_len'],
                                d_word_vec=model_config['d_word_vec'],
                                d_inner=model_config['d_inner'],
                                n_layers=model_config['n_layers'],
                                n_head=model_config['n_head'],
                                dropout=model_config['dropout'],
                                dec_emb_pre_weight_sharing=model_config['dec_emb_pre_weight_sharing'])
        print("loading: ", os.path.join(model_dir, 'last.pth.tar'))
        utils.load_checkpoint(os.path.join(model_dir, 'last.pth.tar'), model)
    else:
        raise Exception

    return model, model_config


def init_measurer():
    file_path = os.path.realpath(__file__)
    base_dir = os.path.dirname(file_path)
    model_dir = os.path.join(base_dir, 'pretrained_models', 'measurer')
    if os.path.exists(os.path.join(model_dir, 'best.pth.tar')):
        with open(os.path.join(model_dir, 'best_config.json')) as f:
            model_config = json.load(f)
            model = Measurer(n_source_vocab=model_config['n_source_vocab'],
                             n_target_vocab=model_config['n_target_vocab'],
                             max_len=model_config['max_len'],
                             d_word_vec=model_config['d_word_vec'],
                             d_inner=model_config['d_inner'],
                             n_layers=model_config['n_layers'],
                             n_head=model_config['n_head'],
                             dropout=model_config['dropout'])
        print("loading: ", os.path.join(model_dir, 'best.pth.tar'))
        utils.load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model)
    elif os.path.exists(os.path.join(model_dir, 'last.pth.tar')):
        with open(os.path.join(model_dir, 'last_config.json')) as f:
            model_config = json.load(f)
            model = Measurer(n_source_vocab=model_config['n_source_vocab'],
                             n_target_vocab=model_config['n_target_vocab'],
                             max_len=model_config['max_len'],
                             d_word_vec=model_config['d_word_vec'],
                             d_inner=model_config['d_inner'],
                             n_layers=model_config['n_layers'],
                             n_head=model_config['n_head'],
                             dropout=model_config['dropout'])
        print("loading: ", os.path.join(model_dir, 'last.pth.tar'))
        utils.load_checkpoint(os.path.join(model_dir, 'last.pth.tar'), model)
    else:
        raise Exception

    return model, model_config


def init_arm_encoder(require_grad=True):
    measurer, measurer_config = init_measurer()
    encoder_config = measurer_config
    encoder_config.pop("n_source_vocab")
    encoder_config["n_vocab"] = encoder_config.pop("n_target_vocab")
    if require_grad:
        measurer.target_encoder.grads()
    else:
        measurer.target_encoder.no_grads()
    return measurer.target_encoder, encoder_config


def init_x86_encoder(require_grad=True):
    measurer, measurer_config = init_measurer()
    encoder_config = measurer_config
    encoder_config.pop("n_target_vocab")
    encoder_config["n_vocab"] = encoder_config.pop("n_source_vocab")
    if require_grad:
        measurer.source_encoder.grads()
    else:
        measurer.source_encoder.no_grads()
    return measurer.source_encoder, encoder_config
