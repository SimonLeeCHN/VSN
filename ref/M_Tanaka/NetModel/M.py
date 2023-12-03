import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CNNBB(nn.Module):
    def __init__(self,input_channels, classes):
        super(CNNBB, self).__init__()
        self.n_channels = input_channels
        self.n_classes = classes
        self.input_channels = input_channels

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels,96,kernel_size=7,stride=2,padding=1),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

            torch.nn.Conv2d(96, 256, kernel_size=5, stride=2),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2,padding=1),

            torch.nn.Conv2d(256, 384, kernel_size=3,stride=1, padding=1),
            # torch.nn.BatchNorm2d(384),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),

            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(384),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),

            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = torch.nn.Sequential(

            torch.nn.Linear(57600,4096),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(),

            # torch.nn.Linear(4096, 4096),
            # # torch.nn.ReLU(inplace=True),
            # # torch.nn.Dropout(0.5),

            torch.nn.Linear(4096,classes),
        )



    def forward(self,x):
        x = self.features(x)
        x = x.view(-1)
        x = self.classifier(x)
        return x

class CNNCP(nn.Module):
    def __init__(self,input_channels, classes):
        super(CNNCP, self).__init__()
        self.n_channels = input_channels
        self.n_classes = classes
        self.input_channels = input_channels

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels,96,kernel_size=5,stride=2,padding=1),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2,stride=1,padding=1),

            torch.nn.Conv2d(96, 256, kernel_size=3, stride=1),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1,padding=1),

            torch.nn.Conv2d(256, 384, kernel_size=3,stride=1, padding=1),
            # torch.nn.BatchNorm2d(384),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),

            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(384),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),

            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = torch.nn.Sequential(

            torch.nn.Linear(6*6*256,4096),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(),

            # torch.nn.Linear(4096, 4096),
            # # torch.nn.ReLU(inplace=True),
            # # torch.nn.Dropout(0.5),

            torch.nn.Linear(4096,classes),
        )



    def forward(self,x):
        x = self.features(x)
        x = x.view(-1)
        x = self.classifier(x)
        return x