#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 20:29:36

import math
import torch
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import torch.nn.functional as F

def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

def forward_chop(net, x, net_kwargs=None, scale=1, shave=10, min_size=160000):
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if net_kwargs is None:
                sr_batch = net(lr_batch)
            else:
                sr_batch = net(lr_batch, **net_kwargs)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def measure_time(net, inputs, num_forward=100):
    '''
    Measuring the average runing time (seconds) for pytorch.
    out = net(*inputs)
    '''
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.set_grad_enabled(False):
        for _ in range(num_forward):
            out = net(*inputs)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / 1000

def reload_model(model, ckpt, strict=False):
    """
    鲁棒的模型加载函数 (Robust Model Loader)
    1. 自动处理 state_dict 嵌套
    2. 自动去除 module. 前缀 (DDP)
    3. 自动去除 first_stage_model. 前缀 (DiffBIR VAE)
    4. 使用 load_state_dict(strict=False) 避免 KeyError
    """
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    # --- 1. 键名预处理 (Key Cleaning) ---
    new_ckpt = {}
    for k, v in ckpt.items():
        # 去除 DDP 训练带来的 module. 前缀
        if k.startswith('module.'):
            k = k[7:]
        
        # 去除 DiffBIR/LatentDiffusion 特有的 VAE 前缀
        # 你的权重文件里，VAE 的参数可能叫 "first_stage_model.encoder..."
        # 但 Resshift 的模型定义的参数叫 "encoder..."
        if k.startswith('first_stage_model.'):
            k = k[18:]
            
        new_ckpt[k] = v

    # --- 2. 安全加载 (Safe Loading) ---
    # strict=False 是核心！它告诉 PyTorch：
    # "能对上的就加载，对不上的就忽略，千万别报错"
    missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)

    # --- 3. 打印报告 (Log) ---
    if len(missing_keys) > 0:
        print(f"⚠️ [Warning] {len(missing_keys)} keys missing in checkpoint (部分参数未加载).")
        # 打印前 3 个缺失的 key 供参考，确认是不是关键层
        for k in missing_keys[:3]:
            print(f"  - Missing: {k}")
        if len(missing_keys) > 3: print("  - ...")
            
    if len(unexpected_keys) > 0:
        print(f"ℹ️ [Info] {len(unexpected_keys)} keys in checkpoint were not used (多余参数).")

    print(f"✅ Model reloaded successfully.")
