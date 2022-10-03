# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = '/home/zolkin/repo/FastPitch/filelists/ru_align/train.txt'
valid_filelist_path = '/home/zolkin/repo/FastPitch/filelists/ru_align/val.txt'
test_filelist_path = '/home/zolkin/repo/FastPitch/filelists/ru_align/test.txt'
cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
n_spks = 12  # 247 for Libri-TTS filelist and 1 for LJSpeech | and 12 for ru-custom
spk_emb_dim = 704  # 64
n_feats = 80
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
cuda = True
log_dir = 'logs/v3'
test_size = 4
n_epochs = 10000
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-6
seed = 42
save_every = 4
out_size = fix_len_compatibility(2 * 22050 // 256)
