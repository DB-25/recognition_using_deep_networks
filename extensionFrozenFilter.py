

import torch
import numpy as np
import cv2
from task1F import loadNetwork

# define Gabor filter parameters
in_channels = 1
out_channels = 10
kernel_size = 5
sigma = 2*np.pi
lambd = 4
psi = 0
gamma = 1

def gabor_filter_weights(in_channels, out_channels, kernel_size, sigma, lambd, psi, gamma):
    # create Gabor filter weights for the first conv layer
    weights = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
    for i in range(out_channels):
        theta = i / out_channels * np.pi
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kernel = np.repeat(kernel[:, :, np.newaxis], in_channels, axis=2)
        weights[i].data = torch.from_numpy(kernel).float()
    return weights


def laplacian_filter_weights(in_channels, out_channels):
    # create Laplacian filter weights for the first conv layer
    weights = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    weights = weights.repeat(out_channels, in_channels, 1, 1)
    return weights

def gaussian_filter_weights(in_channels, out_channels, kernel_size, sigma):
    # create Gaussian filter weights for the first conv layer
    weights = cv2.getGaussianKernel(kernel_size, sigma)
    weights = np.outer(weights, weights)
    weights = np.repeat(weights[:, :, np.newaxis], in_channels*out_channels, axis=2)
    weights = weights.reshape(out_channels, in_channels, kernel_size, kernel_size)
    weights = torch.from_numpy(weights).float()
    return weights



# load the saved models
gabor_model = loadNetwork()
laplacian_model = loadNetwork()
gaussian_model = loadNetwork()

# test the models with some input data
input_data = cv2.imread(r"D:/Users/mohit/MK/PRCV/Project5/recognition_using_deep_networks-master/files/handwrittenDigits/0/0.jpg", cv2.IMREAD_GRAYSCALE)
input_tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).float()

# replace the weights of the first conv layer with Gabor filters and freeze the layer
gabor_weights = gabor_filter_weights(in_channels, out_channels, kernel_size, sigma, lambd, psi, gamma)
gabor_model.conv1.weight.data = gabor_weights
gabor_model.conv1.weight.requires_grad = False

gabor_output = gabor_model(input_tensor)
print("Gabor Filter Model:", gabor_output.argmax().item())

# replace the weights of the first conv layer with Laplacian filters and freeze the layer
laplacian_weights = laplacian_filter_weights(in_channels, out_channels)
laplacian_model.conv1.weight.data = laplacian_weights.float()
laplacian_model.conv1.weight.requires_grad = False

laplacian_output = laplacian_model(input_tensor)
print("Laplacian Filter Model:", laplacian_output.argmax().item())

# replace the weights of the first conv layer with Gaussian filters and freeze the layer
gaussian_weights = gaussian_filter_weights(in_channels, out_channels, kernel_size, sigma)
gaussian_model.conv1.weight.data = gaussian_weights
gaussian_model.conv1.weight.requires_grad = False

gaussian_output = gaussian_model(input_tensor)
print("Gaussian Filter Model:", gaussian_output.argmax().item())









