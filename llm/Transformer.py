import torch
import torch.nn as nn
import math
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000)) / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model // num_head == 0
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None): 
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_score, dim=-1)

        output = torch.matmul(attn_probs, v)
        return output
    
    def split_heads(self, x):   #(batch_size, seq_length, d_model)
        return  rearrange(x, "b s (n d) -> b n s d", n=self.d_k)
    
    def combine_heads(self, x):
        return rearrange(x,"b n s d -> b s (n d)")
    
    def forward(self, q, k, v, mask=None):

        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        attn = self.scaled_dot_product_attention(q, k, v, mask)
        output = self.W_o(self.combine_heads(attn))
        return output



class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    
    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model, dropout):
        super(EncoderLayer, self).__init__()
        self.multi_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out = self.muti_attn()
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
    
class DecoderLayer(nn.Module):

    def __init__(self, d_model, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        maskattn_out = self.self_attn()
        x = self.norm1(x + self.dropout(maskattn_out))
        attn_out = self.cross_attn()
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_out))

