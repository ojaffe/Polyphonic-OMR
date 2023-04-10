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
    def __init__(self, img_channels, layers, no_lstm_layers):
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
                               hidden_size=256,
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
