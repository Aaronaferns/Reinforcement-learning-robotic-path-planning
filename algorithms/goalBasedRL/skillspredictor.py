import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SkillsPredictorModel(nn.Module):
    def __init__(self, nS, layers=[64, 64]):
        super(SkillsPredictorModel, self).__init__()
        modules = []
        input_dim = nS

        for layer_dim in layers:
            modules.append(nn.Linear(input_dim, layer_dim))
            modules.append(nn.GELU())
            input_dim = layer_dim

        modules.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x= self.model(x)
        return F.sigmoid(x)