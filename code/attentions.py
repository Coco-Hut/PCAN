import torch
from torch import nn
import math

# Dot Product Attention
class DotProductAttention(nn.Module):
    
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = self._masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

    # sequence mask
    def _sequence_mask(self,X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    # masked softmax
    def _masked_softmax(self,X, valid_lens):
        
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            
            X = self._sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                                value=-1e6)
            return nn.functional.softmax(X.reshape(shape), dim=-1)
    
# mutihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, value_size, bias=bias)

    def forward(self, queries, keys, values, valid_lens = None):
        queries = self._transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self._transpose_qkv(self.W_k(keys), self.num_heads)
        values = self._transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self._transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    
    def _transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def _transpose_output(self,X, num_heads):
                           
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])    # 1024 8 40 64
        X = X.permute(0, 2, 1, 3)                               # 1024 40 8 64
        return X.reshape(X.shape[0], X.shape[1], -1)            # 1024 40 8*64

class Anomaly_MultiHeadAttention(MultiHeadAttention):
    def __init__(self ,key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout , bias=False,scale = None, **kwargs):
        super(Anomaly_MultiHeadAttention, self).__init__(key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout,bias=False,**kwargs)
        self.scale = scale
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_s = nn.Linear(num_hiddens, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries, keys, values, mode,valid_lens = None,device = "cpu"):
        # mode : main fluctuation all 
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        self.time_lenth = L

        self.distances = torch.zeros((L, L),device = device)
        for i in range(L):
            for j in range(L):
                self.distances[i][j] = abs(i - j)

        H = self.num_heads
        x = queries
        queries = self.W_q(queries).view(B, L, H, -1)
        keys = self.W_k(keys).view(B, S, H, -1)
        values = self.W_v(values).view(B, S, H, -1)
        sigma = self.W_s(x).view(B, L, H)

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L

        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        if device =="cuda":
            prior = prior.cuda()

        if mode == "main":
            prior = 1.0 / (math.sqrt(2 * math.pi) * sigma.detach()) * torch.exp(-prior.detach() ** 2 / 2 / (sigma.detach() ** 2))

        else:
            prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma** 2))
            
        self.prior = prior

        prior =  torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(1, 1, 1, prior.shape[-1])
        
        series = torch.softmax(attn, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", series, values)

        out = V.contiguous()
        if mode == "fluctuation":
            out = out.view(B, L, -1).detach()
        else:
            out = out.view(B, L, -1)

        
        self.series = series
        
        return self.W_o(out)


    def my_kl_loss(self,p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)  # ( batch_size, time_lenth )


    def cal_association_discrepancy(self):
        series_loss = self.my_kl_loss(self.series, \
            (self.prior / torch.unsqueeze(torch.sum(self.prior, dim=-1), dim=-1).repeat(1, 1, 1,self.time_lenth)))
        prior_loss = self.my_kl_loss((self.prior/ torch.unsqueeze(torch.sum(self.prior, dim=-1), dim=-1).repeat(1, 1, 1,self.time_lenth)),\
            self.series.detach())
        AssDis = series_loss + prior_loss
        return AssDis

    @property
    def association_discrepancy(self):
        return self.cal_association_discrepancy()

# feed forward network
class PositionWiseFFN(nn.Module):
    
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# add & Norm
class AddNorm(nn.Module):
    
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)