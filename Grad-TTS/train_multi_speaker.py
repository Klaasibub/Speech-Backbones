# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

# srun -G 4 -c 16 -t 10-0 python3 -m torch.distributed.launch --nproc_per_node 4 train_multi_speaker.py

import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from apex.optimizers import FusedAdam, FusedLAMB

import params
from model import GradTTS
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate, get_sampler
from utils import plot_tensor, save_plot
from text.symbols import symbols


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_spks = params.n_spks
spk_emb_dim = params.spk_emb_dim

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale


def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def init_distributed(args, world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print(f"Initializing distributed training for {rank}")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank)

    # Initialize distributed communication
    dist.init_process_group(
        backend=('nccl' if args.cuda else 'gloo'), init_method='env://',
        rank=rank, world_size=world_size)
    print("Done initializing distributed training")


def log(*args, **kwargs):
    if local_rank == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument("--world_size", type=int, default=os.getenv('WORLD_SIZE', 1))
    args = parser.parse_args()
    local_rank = args.local_rank
    world_size = args.world_size
    distributed_run = world_size > 1

    torch.manual_seed(random_seed + local_rank)
    np.random.seed(random_seed + local_rank)

    if local_rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if distributed_run:
        init_distributed(params, world_size, local_rank)

    log('Initializing logger...')
    if local_rank == 0:
        logger = SummaryWriter(log_dir=log_dir)

    log('Initializing data loaders...')
    train_dataset = TextMelSpeakerDataset(train_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    train_sampler = get_sampler(train_dataset, distributed_run, params.n_spks)

    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        sampler=train_sampler, collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=False)
    test_dataset = TextMelSpeakerDataset(valid_filelist_path, cmudict_path, add_blank,
                                         n_fft, n_feats, sample_rate, hop_length,
                                         win_length, f_min, f_max)

    log('Initializing model...')
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(params, 'cuda', True) else "cpu")
    model = GradTTS(nsymbols, n_spks, spk_emb_dim, n_enc_channels,
                    filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    log('Number of encoder parameters = %.2fm' % (model.encoder.nparams / 1e6))
    log('Number of decoder parameters = %.2fm' % (model.decoder.nparams / 1e6))

    log('Initializing optimizer...')

    optimizer = FusedAdam(
        model.parameters(), lr=params.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=params.weight_decay
    )
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    if distributed_run:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    log('Logging test batch...')
    if local_rank == 0:
        test_batch = test_dataset.sample_test_batch(size=params.test_size)
        for item in test_batch:
            mel, spk = item['y'], item['spk']
            i = int(spk.cpu())
            logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                             global_step=0, dataformats='HWC')
            save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    log('Start training...')
    iteration = 0
    torch.cuda.synchronize()
    for epoch in range(1, n_epochs + 1):
        if distributed_run:
            loader.sampler.set_epoch(epoch)
        if local_rank != 0:
            continue

        model.eval()
        log('Synthesis...')
        with torch.no_grad():
            for item in test_batch:
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                spk = item['spk'].to(torch.long).cuda()
                spk_emb = item['spk_emb'].to(torch.long).cuda()
                i = int(spk.cpu())

                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50, spk_emb=spk_emb)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(),
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(),
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(),
                          f'{log_dir}/alignment_{i}.png')

        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        torch.cuda.synchronize()
        if local_rank == 0:
            iterations = tqdm(loader, total=len(train_dataset) // (batch_size * world_size))
        else:
            iterations = loader
        for batch in iterations:
            model.zero_grad()
            x, x_lengths = to_gpu(batch['x']), to_gpu(batch['x_lengths'])
            y, y_lengths = to_gpu(batch['y']), to_gpu(batch['y_lengths'])
            spk = to_gpu(batch['spk'])
            try:
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     spk=spk, out_size=out_size)
            except AttributeError:
                dur_loss, prior_loss, diff_loss = model.module.compute_loss(x, x_lengths,
                                                                            y, y_lengths,
                                                                            spk=spk, out_size=out_size)
            loss = sum([dur_loss, prior_loss, diff_loss])
            loss.backward()

            try:
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
            except AttributeError:
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.encoder.parameters(), max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(), max_norm=1)

            optimizer.step()
            if local_rank == 0:
                logger.add_scalar('training/duration_loss', dur_loss,
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss,
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss,
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)

                msg = 'Epoch: {}, iteration: {:.3f} | dur_loss: {:.3f}, prior_loss: {:.3f}, diff_loss: {:.3f}'
                msg = msg.format(epoch, iteration, dur_loss.item(), prior_loss.item(), diff_loss.item())
                iterations.set_description(msg)

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
            iteration += 1

        if local_rank == 0:
            msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
            msg += '| prior loss = %.3f ' % np.mean(prior_losses)
            msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
            with open(f'{log_dir}/train.log', 'a') as f:
                f.write(msg)

            if epoch % params.save_every > 0:
                continue

            ckpt = model.state_dict()
            torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
