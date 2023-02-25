from torch import nn
from embeddings import Embeddings
from transformer_encoder_layer import TransformerEncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x
