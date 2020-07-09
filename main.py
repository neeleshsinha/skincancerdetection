import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import pretrainedmodels

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
       super(SEResNext50_32x4d, self).__init__()
       self.model = pretrainedmodels.__dict__("se_ResNet50_32x4d")