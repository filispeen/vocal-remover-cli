import argparse
from datetime import datetime
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from lib import dataset
from lib import nets
from lib import spec_utils


def setup_logger(name, logfile='LOGFILENAME.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(logfile, encoding='utf8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def train_epoch(dataloader, model, device, optimizer, accumulation_steps):
    model.train()
    sum_loss = 0
    crit = nn.L1Loss()

    for itr, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred, aux = model(X_batch)

        loss_main = crit(pred * X_batch, y_batch)
        loss_aux = crit(aux * X_batch, y_batch)

        loss = loss_main * 0.8 + loss_aux * 0.2
        accum_loss = loss / accumulation_steps
        accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        sum_loss += loss.item() * len(X_batch)

    # the rest batch
    if (itr + 1) % accumulation_steps != 0:
        optimizer.step()
        model.zero_grad()

    return sum_loss / len(dataloader.dataset)


def validate_epoch(dataloader, model, device):
    model.eval()
    sum_loss = 0
    crit = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model.predict(X_batch)

            y_batch = spec_utils.crop_center(y_batch, pred)
            loss = crit(pred, y_batch)

            sum_loss += loss.item() * len(X_batch)

    return sum_loss / len(dataloader.dataset)


def train(logger, timestamp, a_seed, a_val_filelist, a_dataset, a_split_mode, a_val_rate, a_debug, a_n_fft, a_pretrained_model, a_gpu, a_learning_rate, a_lr_decay_factor, a_lr_decay_patience, a_lr_min, a_sr, a_reduction_level, a_hop_length, a_patches, a_cropsize, a_reduction_rate, a_mixup_rate, a_mixup_alpha, a_batchsize, a_num_workers, a_val_cropsize, a_val_batchsize, a_epoch, a_accumulation_steps):
    random.seed(a_seed)
    np.random.seed(a_seed)
    torch.manual_seed(a_seed)

    val_filelist = []
    if a_val_filelist is not None:
        with open(a_val_filelist, 'r', encoding='utf8') as f:
            val_filelist = json.load(f)

    train_filelist, val_filelist = dataset.train_val_split(
        dataset_dir=a_dataset,
        split_mode=a_split_mode,
        val_rate=a_val_rate,
        val_filelist=val_filelist
    )

    if a_debug:
        logger.info('### DEBUG MODE')
        train_filelist = train_filelist[:1]
        val_filelist = val_filelist[:1]
    elif a_val_filelist is None and a_split_mode == 'random':
        with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        logger.info('{} {} {}'.format(i + 1, os.path.basename(X_fname), os.path.basename(y_fname)))

    device = torch.device('cpu')
    model = nets.CascadedNet(a_n_fft, 32, 128)
    if a_pretrained_model is not None:
        model.load_state_dict(torch.load(a_pretrained_model, map_location=device))
    if torch.cuda.is_available() and a_gpu >= 0:
        device = torch.device('cuda:{}'.format(a_gpu))
        model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=a_learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=a_lr_decay_factor,
        patience=a_lr_decay_patience,
        threshold=1e-6,
        min_lr=a_lr_min,
        verbose=True
    )

    bins = a_n_fft // 2 + 1
    freq_to_bin = 2 * bins / a_sr
    unstable_bins = int(200 * freq_to_bin)
    stable_bins = int(22050 * freq_to_bin)
    reduction_weight = np.concatenate([
        np.linspace(0, 1, unstable_bins, dtype=np.float32)[:, None],
        np.linspace(1, 0, stable_bins - unstable_bins, dtype=np.float32)[:, None],
        np.zeros((bins - stable_bins, 1), dtype=np.float32),
    ], axis=0) * a_reduction_level

    training_set = dataset.make_training_set(
        filelist=train_filelist,
        sr=a_sr,
        hop_length=a_hop_length,
        n_fft=a_n_fft
    )

    train_dataset = dataset.VocalRemoverTrainingSet(
        training_set * a_patches,
        cropsize=a_cropsize,
        reduction_rate=a_reduction_rate,
        reduction_weight=reduction_weight,
        mixup_rate=a_mixup_rate,
        mixup_alpha=a_mixup_alpha
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=a_batchsize,
        shuffle=True,
        num_workers=a_num_workers
    )

    patch_list = dataset.make_validation_set(
        filelist=val_filelist,
        cropsize=a_val_cropsize,
        sr=a_sr,
        hop_length=a_hop_length,
        n_fft=a_n_fft,
        offset=model.offset
    )

    val_dataset = dataset.VocalRemoverValidationSet(
        patch_list=patch_list
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=a_val_batchsize,
        shuffle=False,
        num_workers=a_num_workers
    )

    log = []
    best_loss = np.inf
    for epoch in range(a_epoch):
        logger.info('# epoch {}'.format(epoch))
        train_loss = train_epoch(train_dataloader, model, device, optimizer, a_accumulation_steps)
        val_loss = validate_epoch(val_dataloader, model, device)

        logger.info(
            '  * training loss = {:.6f}, validation loss = {:.6f}'
            .format(train_loss, val_loss)
        )

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            logger.info('  * best validation loss')
            model_path = 'models/model_iter{}.pth'.format(epoch)
            torch.save(model.state_dict(), model_path)

        log.append([train_loss, val_loss])
        with open('loss_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(log, f, ensure_ascii=False)

def train_start(a_seed, a_val_filelist, a_dataset, a_split_mode, a_val_rate, a_debug, a_n_fft, a_pretrained_model, a_gpu, a_learning_rate, a_lr_decay_factor, a_lr_decay_patience, a_lr_min, a_sr, a_reduction_level, a_hop_length, a_patches, a_cropsize, a_reduction_rate, a_mixup_rate, a_mixup_alpha, a_batchsize, a_num_workers, a_val_cropsize, a_val_batchsize, a_epoch, a_accumulation_steps):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        train(logger, timestamp, a_seed, a_val_filelist, a_dataset, a_split_mode, a_val_rate, a_debug, a_n_fft, a_pretrained_model, a_gpu, a_learning_rate, a_lr_decay_factor, a_lr_decay_patience, a_lr_min, a_sr, a_reduction_level, a_hop_length, a_patches, a_cropsize, a_reduction_rate, a_mixup_rate, a_mixup_alpha, a_batchsize, a_num_workers, a_val_cropsize, a_val_batchsize, a_epoch, a_accumulation_steps)
    except Exception as e:
        logger.exception(e)

        
#if __name__ == '__main__':
#    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
#    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))
#
#    try:
#        main()
#    except Exception as e:
#        logger.exception(e)
