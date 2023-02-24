import os
import numpy as np
import copy
import math
import random
from typing import Optional, Tuple
from dataclasses import dataclass, field
from itertools import chain

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Projection(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Projection,self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x

class LinearELU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LinearELU,self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.elu = nn.ELU()
    def forward(self, x):
        x = self.linear(x)
        x = self.elu(x)
        return x

class SequenceMask(nn.Module):
    def __init__(self, dtype=torch.bool):
        super(SequenceMask,self).__init__()
        self.dtype=dtype
    def forward(self, lengths, maxlen=None):
        lengths = torch.ones(lengths,dtype=self.dtype)
        maxlen = lengths.max() if maxlen is None else maxlen
        row_vector = torch.arange(0, maxlen, 1) #device=self.device
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = (row_vector < matrix).to(dtype=self.dtype) #device=self.device

        return mask