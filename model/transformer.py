import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# import lightning as L
from torch import optim, utils, Tensor


class MultiHeadedAttention(nn.Module):
    def __init__(self, embedding_size, n_heads, dropout = 0.0, bias=True):
        super().__init__()
        assert embedding_size %  n_heads == 0  ## simplifying to keep qkv_size fixed to embedding_size/n_heads, no theoretical reason for this except, but probably good for efficiency of operations. 
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.qkv_size = embedding_size // n_heads
        self.q_proj = nn.Linear(self.embedding_size, self.qkv_size*self.n_heads, bias=bias)
        self.k_proj = nn.Linear(self.embedding_size, self.qkv_size*self.n_heads, bias=bias)
        self.v_proj = nn.Linear(self.embedding_size, self.qkv_size*self.n_heads, bias=bias)
        self.out_proj = nn.Linear(self.embedding_size, self.embedding_size, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
    def forward(self, embedding, mask = None):
        batch, seq_len, emb_size = embedding.shape
        q = self.q_proj(embedding).reshape([batch, seq_len, self.n_heads, self.qkv_size]).transpose(1,2) # [batch, n_heads, seq_len, qkv_size]
        k = self.k_proj(embedding).reshape([batch, seq_len, self.n_heads, self.qkv_size]).transpose(1,2) # [batch, n_heads, seq_len, qkv_size]
        v = self.v_proj(embedding).reshape([batch, seq_len, self.n_heads, self.qkv_size]).transpose(1,2) # [batch, n_heads, seq_len, qkv_size]
        # attn = self.attention(embedding, mask) # result should be [batch, n_heads, seq_len, seq_len]
        # attended_v = torch.matmul(attn, v).transpose(1,2) # [batch, n_heads, seq_len, qkv_size]
        # out = self.out_dropout(self.out_proj(attended_v.reshape([batch, seq_len, emb_size])))
        if mask is not None and len(mask.shape) == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        attended_v = F.scaled_dot_product_attention(q, k, v, attn_mask = mask, dropout_p = self.attn_dropout.p)
        out = self.out_dropout(self.out_proj(attended_v.reshape([batch, seq_len, emb_size])))
        return out
    def attention(self, embedding, mask = None):
        batch, seq_len, emb_size = embedding.shape
        q = self.q_proj(embedding).reshape([batch, seq_len, self.n_heads, self.qkv_size]).transpose(1,2) # [batch, n_heads, seq_len, qkv_size]
        k = self.k_proj(embedding).reshape([batch, seq_len, self.n_heads, self.qkv_size]).transpose(1,2) # [batch, n_heads, seq_len, qkv_size]
        # v = self.v_proj(embedding).reshape([batch, seq_len, self.n_heads, self.qkv_size]).transpose(1,2) # [batch, n_heads, seq_len, qkv_size]
        score = torch.matmul(q, k.transpose(-2,-1)) * (1.0/math.sqrt(self.qkv_size))
        if mask is not None:
            score = score + mask.unsqueeze(1).unsqueeze(1)
        attn = self.attn_dropout(torch.softmax(score, dim=-1)) # result should be [batch, n_heads, seq_len, seq_len]
        return attn


class MLPLayer(nn.Module):
    def __init__(self, embedding_size, bias=True, dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.bias = bias
        self.embedding_size = embedding_size
        self.activation = nn.GELU()
        self.expand = nn.Linear(embedding_size, 2*embedding_size, bias=bias)
        self.contract = nn.Linear(2*embedding_size, embedding_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.expand(x)
        x = self.activation(x)
        x = self.contract(x)
        x = self.dropout(x)
        return x
    


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, n_heads, mha_dropout = 0.0, mha_bias=True, mlp_dropout = 0.0, mlp_bias=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.MHAttn = MultiHeadedAttention(embedding_size, n_heads, mha_dropout, mha_bias)
        self.MLP = MLPLayer(embedding_size, bias=mlp_bias, dropout = mlp_dropout)
        self.PreLNAttention = nn.LayerNorm(embedding_size)
        self.PreLNMLP = nn.LayerNorm(embedding_size)
    def forward(self, embedding, mask = None):
        x1 = self.PreLNAttention(embedding)
        x1 = self.MHAttn(x1, mask=mask)
        x1 = embedding + x1
        x2 = self.PreLNMLP(x1)
        x2 = self.MLP(x2)
        return x1 + x2

class MaskedSequential(nn.Sequential):
    def forward(self, x, mask):
        for mod in self._modules.values():
            x = mod(x, mask)
        return x



class Transformer(nn.Module):
    def __init__(self, pos_enc, config):
        super().__init__()
        self.pos_enc = pos_enc
        thl = []
        for _ in range(config["n_trans_blocks"]):
            thl.append(TransformerBlock(**config['tb']))
        self.transformer = MaskedSequential(*thl)
        self.cls = config['cls']
        if self.cls:
            self.cls_token = nn.Parameter(torch.randn((1,1,config['input_size']))) # think about whether this should require grad? it seems the ViT from jax adds it to params as well. 
        self.in_proj = nn.Sequential(nn.Linear(config["input_size"], config['tb']["embedding_size"],bias=True), nn.GELU())
        self.out_proj = nn.Sequential(nn.Linear(config['tb']['embedding_size'], 1, bias=True))
        self.FinalLN = nn.LayerNorm(config['tb']['embedding_size'])
        self.transformer.apply(self._init_weights) ## apply non-standard initializations here


    ## Adapted from nanoGPT/Karpathy's videos. I think the std might be too low for shallow models, but should test empirically
    ## The bias initialization is probably the more imporant step here
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0/(module.in_features**0.5))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, mask = None, coords = None):
        
        
        if self.cls:
            cls_tokens = self.cls_token.expand([x.shape[0], 1, x.shape[2]]) # create a copy for each element in the batch
            x = torch.cat((cls_tokens,x), dim=1) ## add them to the sequence 
            if mask is not None: 
                mask = torch.cat((torch.zeros([x.shape[0],1], dtype=torch.bool),mask), dim=1)
        

        ## Note, it seems standard to add a position embedding to the CLS token. The only reason I can see to do this is so that it is distributed
        ## similarly to the other tokens, but we are just adding a parameter to a parameter here.

        ## I am making an assumption that coords here is compatible with the position encoding, and that it is already going to produce embeddings into the right dimension.
        if self.pos_enc:
            pos_encoded = self.pos_enc(coords)
            x = x + pos_encoded
        
        x = self.in_proj(x)
        x = self.transformer(x, mask=mask)
        if self.cls:
            x = x[:,0,:]
        else:
            x = torch.mean(x, dim=1)
        y = self.out_proj(self.FinalLN(x))
        return(y)

## from transformer_from_scratch.model.transformer import Transformer


## Config expects: n_trans_blocks, tb.n_heads, tb.mha_dropout, tb.mlp_dropout, tb.mlp_bias, tb.mha_bias, tb.embedding_size
# config = {
#     "n_trans_blocks": 2,
#     "input_size": 768,
#     "tb": {
#         "n_heads": 8,
#         "embedding_size": 512,
#         "mlp_dropout": 0.0,
#         "mlp_bias": True,
#         "mha_dropout": 0.0,
#         "mha_bias": True
#     },
#     "cls": True
# }
# test = Transformer(False, config)
# import torch
# x = torch.randn((2,10,768))
# test(x, None)

# mask = torch.zeros((2,10))
# mask[0,5:] = -torch.inf
# test(x, mask=mask)