## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torch.autograd import Variable

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
#         self.fc1 = torch.nn.Linear(64 * 53 * 53, 1024)
#         self.fc2 = torch.nn.Linear(1024, 256)
#         self.fc3 = torch.nn.Linear(256, 136)
        self.fc1 = torch.nn.Linear(64 * 53 * 53, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, input):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        layer1 = torch.functional.F.relu(self.conv1(input))
        layer2 = self.pool1(layer1)
        layer3 = torch.functional.F.relu(self.conv2(layer2))
        layer4 = self.pool2(layer3)
        layer5 = layer4.view(-1, 64 * 53 * 53)
        output = self.fc1(layer5)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return output
