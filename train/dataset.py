import os
import cv2
import ast

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

from make_csv import make_csv
from utils import resize


class PolyphonicDataset(Dataset):
    def __init__(self, data_cfg: dict, note2idx: dict, idx2note: dict, vocab_size: int):

        self.data = pd.read_csv(data_cfg.get("csv_out", None))
        self.data_len = len(self.data)

        self.data_cfg = data_cfg

        self.note2idx = note2idx
        self.idx2note = idx2note
        self.vocab_size = vocab_size

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path, notes = self.data.iloc[idx]

        notes = ast.literal_eval(notes)

        # Deal with alpha (transparent PNG) - POLYPHONIC DATASET IMAGES
        sample_img = cv2.imread(os.path.join(self.data_cfg.get("data_dir", None), img_path), cv2.IMREAD_UNCHANGED)
        try:
            if sample_img.shape[2] == 4:     # we have an alpha channel
                a1 = ~sample_img[:,:,3]        # extract and invert that alpha
                sample_img = cv2.add(cv2.merge([a1,a1,a1,a1]), sample_img)   # add up values (with clipping)
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2RGB)    # strip alpha channel
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)
            elif sample_img.shape[2] == 3:   # no alpha channel (musicma_abaro)
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY) 
        except IndexError: # 2d image
            pass

        height = self.data_cfg.get("img_height", 128)
        sample_img = resize(sample_img, height, 880)

        tokens = torch.tensor([self.note2idx[x] for x in notes])

        return torch.tensor(sample_img, dtype=torch.float32).unsqueeze(0), tokens


class PadCollate:
    def __init__(self, PAD_IDX):
        self.PAD_IDX = PAD_IDX

    def __call__(self, batch):
        imgs, tgt = zip(*batch)

        imgs = torch.stack(list(imgs), dim=0)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=self.PAD_IDX)

         # Construct padding mask
        tgt_pad_mask = tgt.ne(self.PAD_IDX)

        return imgs, tgt, tgt_pad_mask


def create_tokenizer(data_cfg):
    SOS_IDX = 0
    EOS_IDX = 1
    PAD_IDX = 2
    DOT_IDX = 3
    note2idx = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<DOT>": 3, "<CHORD START>": 4, "<CHORD END>": 5}
    idx2note = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<DOT>", 4: "<CHORD START>", 5: "<CHORD END>"}
    initial_len = len(note2idx)

    with open(data_cfg.get("vocab_path", None), 'r') as f:
        words = f.read().split('\n')

    for i, word in enumerate(words):
        idx = i + initial_len  # Offset by no. special tokens
        note2idx[word] = idx
        idx2note[idx] = word
        
    vocab_size = len(note2idx)
    return note2idx, idx2note, vocab_size, SOS_IDX, EOS_IDX, PAD_IDX, DOT_IDX


def load_data(data_cfg):
    note2idx, idx2note, vocab_size, SOS_IDX, EOS_IDX, PAD_IDX, DOT_IDX = create_tokenizer(data_cfg)

    batch_size = data_cfg.get("batch_size", 4)

    # Create dataset from cleaned data, if doesn't already exist
    csv_out = data_cfg.get("csv_out", None)
    if not os.path.isfile(csv_out) or data_cfg.get("remake_csv", False):
        make_csv(data_cfg)

    dataset = PolyphonicDataset(data_cfg, note2idx, idx2note, vocab_size)

    # Create splits
    indices = list(range(len(dataset)))
    if data_cfg.get("shuffle", True):
        np.random.shuffle(indices)

    train_prop, val_prop, test_prop = data_cfg.get("dataset_split", [.8, .1, .1])
    train_split = int(np.floor(train_prop * len(dataset)))
    val_split = train_split + int(np.floor(val_prop * len(dataset)))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), 
                                                    collate_fn=PadCollate(PAD_IDX))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), 
                                                    collate_fn=PadCollate(PAD_IDX))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), 
                                                    collate_fn=PadCollate(PAD_IDX))

    return train_loader, val_loader, test_loader, note2idx, idx2note, vocab_size, SOS_IDX, EOS_IDX, PAD_IDX