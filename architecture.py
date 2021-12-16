import torch
import torch.nn as nn
import torch.nn.functional as F


class architectureMNIST(nn.Module):

    def __init__(self,dropout_layer):

        super(architectureMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1,32,3)
        
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32,10,3)
 
        self.flat = nn.Flatten()
 
        self.batchnorm = nn.BatchNorm1d(1210)

        self.dropout = dropout_layer

        self.classification = nn.Linear(1210,10)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.classification(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class architectureCIFAR10(nn.Module):

    def __init__(self,dropout_layer):

        super(architectureCIFAR10, self).__init__()

        self.conv1 = nn.Conv2d(3,32,3)
        
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32,10,3)
 
        self.flat = nn.Flatten()
 
        self.batchnorm = nn.BatchNorm1d(1690)

        self.dropout = dropout_layer

        self.classification = nn.Linear(1690,10)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.classification(x)

        return x
