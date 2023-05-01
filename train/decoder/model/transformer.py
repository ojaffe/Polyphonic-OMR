"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from decoder.model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.device = device

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=vocab_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, im_emb, tgt, tgt_pad_mask, tgt_causal_mask):
        output = self.decoder(im_emb, tgt, tgt_pad_mask, tgt_causal_mask)
        return output
