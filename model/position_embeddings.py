from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# import lightning as L
from torch import optim, utils, Tensor

## The use of nn.F.interpolate is inspired by the note in the DINOv2 repository

class AbstractPositionalEmbedding2D(nn.Module):
    @abstractmethod
    def __init__(self, height, width, embedding_dim, cls_token=False):
        super().__init__()
        self.cls_token = cls_token
        self.num_patches = height * width
        self.embedding_dim = embedding_dim
        self.height = height
        self.width = width
        self.num_extra_tokens = 1 if self.cls_token else 0  # here in case I want to play around with adding more than 1 token later.
    ## coords is a tensor of shape (batch_size, num_tokens, 2)
    def forward(self, coords):
        # assert type(coords) == torch.Tensor
        cls_embedding = self.pos_embed[:, :self.num_extra_tokens, :]
        batch_size, num_tokens_coords, _ = coords.shape
        pos_embedding = self.pos_embed[:, self.num_extra_tokens:, :]
        if num_tokens_coords == self.height * self.width:
            pos_embedding = pos_embedding.reshape(self.height, self.width, self.embedding_dim)
            pos_embedding = pos_embedding[coords[:, :, 0], coords[:, :, 1], :]
            return torch.cat((cls_embedding.expand([pos_embedding.shape[0], 1, self.embedding_dim]),pos_embedding), dim=1) # batch, seq_len, embedding_dim
        else: # now we handle the case with resizing
            pos_embedding = pos_embedding.reshape(1,self.height, self.width, self.embedding_dim).permute(0,3, 1, 2)
            target_height = coords[:, :, 0].max() - coords[:, :, 0].min() + 1
            target_width = coords[:, :, 1].max() - coords[:, :, 1].min() + 1
            pos_embedding = F.interpolate(pos_embedding, size=(target_height, target_width), mode='bicubic')
            pos_embedding = pos_embedding.permute(0, 2, 3, 1).reshape(target_height, target_width, self.embedding_dim)
            pos_embedding = pos_embedding[coords[:, :, 0], coords[:, :, 1], :]
            return torch.cat((cls_embedding.expand([pos_embedding.shape[0], 1, self.embedding_dim]),pos_embedding), dim=1)


class TrainedPositionalEmbedding2D(AbstractPositionalEmbedding2D):
    def __init__(self, height, width, embedding_dim, cls_token=False):
        """
        Initialize the TrainedPositionalEmbedding2D class.

        Parameters
        ----------
        height : int
            The height of the 2D grid of patches.
        width : int
            The width of the 2D grid of patches.
        embedding_dim : int
            The dimensionality of the embeddings.
        cls_token : bool, optional
            Whether to include a cls token, by default False
        """
        super().__init__(height, width, embedding_dim, cls_token=cls_token)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + self.num_extra_tokens, self.embedding_dim))
        self.init_weights()
    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


class SinePositionalEmbedding2D(AbstractPositionalEmbedding2D):
    def __init__(self, height, width, embedding_dim, cls_token=False):
        """
        Initialize the SinePositionalEmbedding2D class.

        Parameters
        ----------
        height : int
            The height of the 2D grid of patches.
        width : int
            The width of the 2D grid of patches.
        embedding_dim : int
            The dimensionality of the embeddings.
        cls_token : bool, optional
            Whether to include a cls token, by default False
        """
        super().__init__(height, width, embedding_dim, cls_token=cls_token)
        self.pos_embed = self.generate_sine_embedding(self.num_patches + self.num_extra_tokens, self.embedding_dim)
    def generate_sine_embedding(self, num_patches, embedding_dim):
        pe = torch.zeros(1, num_patches, embedding_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe