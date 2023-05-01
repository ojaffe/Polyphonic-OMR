"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from decoder.layers.layer_norm import LayerNorm
from decoder.layers.multi_head_attention import MultiHeadAttention
from decoder.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        #self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        #self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.enc_dec_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, enc, dec, dec_pad_mask, dec_causal_mask):
        # 1. compute self attention
        _x = dec

        x, _ = self.self_attention(query=dec, key=dec, value=dec, key_padding_mask=dec_pad_mask, is_causal=True)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x, _ = self.enc_dec_attention(query=x, key=enc, value=enc)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
