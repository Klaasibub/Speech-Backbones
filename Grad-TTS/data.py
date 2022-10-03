# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        return (text, mel)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path=None, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path) if cmudict_path else None
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, speaker_emb, text, speaker = line[0], line[-3], line[-2], line[-1]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker_emb = torch.load(speaker_emb)
        speaker = self.get_speaker(speaker)
        return (text, mel, speaker, speaker_emb)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker, speaker_emb = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker, 'spk_emb': speaker_emb}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []
        spk_emb = []

        for i, item in enumerate(batch):
            y_, x_, spk_, spk_emb_ = item['y'], item['x'], item['spk'], item['spk_emb']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)
            spk_emb.append(spk_emb_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        spk_emb = torch.cat(spk_emb, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk, 'spk_emb': spk_emb}


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over Sampler for distributed training. It allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with torch.nn.parallel.DistributedDataParallel. In such a case, each
    process can pass a torch.utils.data.DistributedSampler instance as a torch.utils.data.DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note:
        Dataset is assumed to be of constant size.

    Args:
        sampler: Sampler used for subsampling.
        num_replicas (int, optional): Number of processes participating in distributed training. By default,
            world_size is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within num_replicas. By default, rank is retrieved
            from the current distributed group.
        shuffle (bool, optional): If True, sampler will shuffle the indices. Default: True.
        seed (int, optional): random seed used to shuffle the sampler if shuffle=True. This number should be
            identical across all processes in the distributed group. Default: 0.

    Reference: https://github.com/pytorch/pytorch/issues/23430

    """

    def __init__(
        self,
        sampler,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

    def __iter__(self):
        indices = list(self.dataset)[: self.total_size]

        # Add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size, f"{len(indices)} != {self.total_size}"

        # Subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples, f"{len(indices)} != {self.num_samples}"

        return iter(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        elif hasattr(self.dataset, "generator"):
            self.dataset.generator = torch.Generator().manual_seed(self.seed + epoch)

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)


def get_speaker_balancer_weights(items: list):
    speaker_names = np.array([item[-1] for item in items])
    unique_speaker_names = np.unique(speaker_names).tolist()
    speaker_ids = [unique_speaker_names.index(speaker) for speaker in speaker_names]
    speaker_count = np.array([len(np.where(speaker_names == speaker)[0]) for speaker in unique_speaker_names])
    weight_speaker = 1.0 / speaker_count
    dataset_samples_weight = np.array([weight_speaker[speaker_id] for speaker_id in speaker_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()


def get_sampler(dataset, distributed_run, n_speakers, speaker_weighted_sampler_alpha=1.0):
    weights = None

    if n_speakers:
        alpha = speaker_weighted_sampler_alpha
        if weights is not None:
            weights += get_speaker_balancer_weights(dataset.filelist) * alpha
        else:
            weights = get_speaker_balancer_weights(dataset.filelist) * alpha

    # sampler for DDP
    sampler = WeightedRandomSampler(weights, len(weights))
    sampler = DistributedSamplerWrapper(sampler) if distributed_run else sampler

    return sampler
