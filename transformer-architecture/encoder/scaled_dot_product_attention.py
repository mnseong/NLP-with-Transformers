import torch
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1,2))/sqrt(dim_k)
    # masking step
    if mask is not None:
        scores = scores.masked_fill(mask==0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)