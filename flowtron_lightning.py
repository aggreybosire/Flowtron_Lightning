import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from flowtron import Flowtron

class flowtronLighting(pl.LightningModule):
    def __init__(self):
        super(Flowtron, self).__init__()
        self.flowtron = Flowtron()

model = flowtronLighting()