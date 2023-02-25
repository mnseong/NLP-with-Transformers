from torch import nn
from attention_head import MultiHeadAttention
from feed_forward import FeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)  # normalization layers
        x = x + self.attention(hidden_state)  # attention에 skip 연결 적용
        x = x + self.feed_forward(self.layer_norm_2(x))  # skip 연결과 feed forward layer 적용
        return x
