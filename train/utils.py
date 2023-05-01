import cv2
import random
import yaml

from tqdm import tqdm
import numpy as np

import torch


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    path: path to YAML configuration file
    return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    return cfg


def normalize(image):
    """
    Makes black pixels of image take on high value, white pixels
    take on low value
    """
    return (255. - image)/255.


def resize(image, height, width=None):
    """
    Resizes an image to desired width and height
    """

    # Have width be original width
    if width is None:
        width = int(float(height * image.shape[1]) / image.shape[0])

    sample_img = cv2.resize(image, (width, height))
    return sample_img


def greedy_decode(encoder, decoder, data_loader, PAD_IDX, SOS_IDX, EOS_IDX, device):
    """
    Decode single example from test dataloader greedily
    """
    tgt_input = [[SOS_IDX]]
    tgt_input = torch.tensor(tgt_input).to(device)

    # Get example
    batch = next(iter(data_loader))
    image, tgt = batch
    
    image, tgt = image[0].unsqueeze(0).to(device), tgt[0].unsqueeze(0).to(device)

    with torch.no_grad():
        im_emb = encoder(image)
        for i in tqdm(range(200)):
            output = decoder(im_emb, tgt_input, None, None)

            final_out = output[:, output.shape[1]-1, :]
            top_token = torch.argmax(final_out).item()
                
            # Add decoded to next input
            decoded_example = torch.tensor([[top_token]]).to(device)
            tgt_input = torch.concatenate((tgt_input, decoded_example), dim=1)

            if top_token == EOS_IDX:
                break

    return tgt[0].tolist(), tgt_input.detach().tolist()[0]


def build_pad_mask(inputs, pad_idx):
    pad_mask = (inputs == pad_idx).bool()

    return pad_mask


def build_causal_mask(inputs, num_heads):
    batch_size, sequence_length = inputs.size()

    causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=inputs.device))
    causal_mask = (causal_mask == 0)
    causal_mask = causal_mask.repeat(batch_size * num_heads, 1, 1).bool()
    return causal_mask


def calc_acc(input, target):
    """
    :param input: [B, T, V]
    :param target: [B, T]
    """
    input = input.softmax(-1)
    categorical_input = input.argmax(-1)

    bool_acc = categorical_input.long() == target.long()
    return bool_acc.sum().to(torch.float) / bool_acc.numel()
