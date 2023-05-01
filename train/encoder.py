import torch.nn as nn
import torch
from einops import rearrange
import math


# Building block introducted in ResNet paper https://arxiv.org/abs/1512.03385
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()

        self.downsample = downsample
        
        self.bn1 = nn.BatchNorm2d(out_channels)  # https://arxiv.org/abs/1502.03167
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.2)
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        
        out = self.relu(out)
        out = self.drop(out)
        
        return out


class EncoderCRNN(nn.Module):
    def __init__(self, img_channels, layers, no_lstm_layers, lstm_hidden):
        super(EncoderCRNN, self).__init__()
                
        # Initial conv layer to increase channels
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False  # Bias false since we're applying batch norm after
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels)  # https://arxiv.org/abs/1502.03167
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(p=0.2)
        
        # Compress input size, extract relevant features
        block = Block
        self.l1 = self._construct_layer(block, 64, layers[0])
        self.l2 = self._construct_layer(block, 128, layers[1], stride=2)
        self.l3 = self._construct_layer(block, 256, layers[2], stride=2)
        self.l4 = self._construct_layer(block, 512, layers[3], stride=2)

        # Bi-LSTM encoding
        self.bi_lstm = nn.LSTM(input_size=512,
                               hidden_size=int(lstm_hidden / 2),
                               num_layers=no_lstm_layers,
                               bias=False,
                               batch_first=True,
                               bidirectional=True)
        
    
    def _construct_layer(self, block, out_channels, blocks, stride=1):
        """
        block: object to construct nn block from
        out_channels: no. channels to expand to
        blocks: no. blocks to construct for layer
        """
        downsample = None
        
        # Downsample on all layers apart from first
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)  # Combine layers into single module
    
    
    def forward(self, x):
        x = self.conv1(x)  # (b, 1, 128, 880) -> (b, 64, 64, 440)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (b, 64, 64, 440) -> (b, 64, 32, 220)
        
        x = self.l1(x)
        x = self.l2(x)  # (b, 64, 32, 220) -> (b, 128, 16, 110)
        x = self.l3(x)  # (b, 128, 16, 110) -> (b, 256, 8, 55)
        x = self.l4(x)  # (b, 256, 8, 55) -> (b, 512, 4, 28)

        x = torch.reshape(x, (x.shape[0], 
                              x.shape[1], 
                              x.shape[2] * x.shape[3]))  # (b, 512, 4, 28) -> (b, 512, 112)
        x = rearrange(x, "b d w -> b w d")  # (b, 512, 112) -> (b, 112, 512)

        x, (h, c) = self.bi_lstm(x)  # (b, 112, 512) -> (b, 112, 1024)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super(Decoder, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout=drop_prob)
        self.encoder = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=ffn_hidden,
                                                   dropout=drop_prob, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, im_emb, tgt_input, tgt_pad_mask, tgt_causal_mask):
        emb = self.encoder(tgt_input)
        if not self.train() and tgt_causal_mask is not None:
            tgt_causal_mask = tgt_causal_mask[0]
            
        output = self.decoder(emb, im_emb, tgt_mask=tgt_causal_mask, tgt_key_padding_mask=tgt_pad_mask)
        output = self.lm_head(output)

        return output
