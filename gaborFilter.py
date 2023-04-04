import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import cv2
from main import NeuralNetwork
from task1F import loadNetwork



class GaborFilterInitializer:
    def __init__(self, in_channels, out_channels, kernel_size, sigma, lambd, psi, gamma):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lambd = lambd
        self.psi = psi
        self.gamma = gamma

    def __call__(self, weights):
        for i in range(self.out_channels):
            theta = i / self.out_channels * np.pi
            kernel = cv2.getGaborKernel((self.kernel_size, self.kernel_size), self.sigma, theta, self.lambd, self.gamma, self.psi, ktype=cv2.CV_32F)
            kernel = np.repeat(kernel[:, :, np.newaxis], self.in_channels, axis=2)
            weights[i].data = torch.from_numpy(kernel).float()
        return weights

# load the saved model
model = loadNetwork()

# replace the weights of the first conv layer with Gabor filters and freeze the layer
gabor_initializer = GaborFilterInitializer(in_channels=1, out_channels=10, kernel_size=5, sigma=2*np.pi, lambd=4, psi=0, gamma=1)
model.conv1.apply(gabor_initializer)
model.conv1.weight.requires_grad = False

# test the model with some input data
input_data = cv2.imread(f"DB-25/recognition_using_deep_networks/files/handwrittenDigits/0/0.jpg")
output = model(input_data)
print(output)
