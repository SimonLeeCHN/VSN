import torch
from thop import profile, clever_format
from torchinfo import summary
from NetModel import *

model = VSNLite1MD(n_channels=3, n_classes=1)

# pytorch-OpCounter
# input = torch.randn(1, 3, 1000, 520)
# ops, params = profile(model, inputs=(input,))
# print("Params(M): ", params/(1000 ** 2))
# print("FLOPs(G): ", ops/(1000 ** 3))

# torchinfo
summary(model, (1, 3, 1000, 520))
