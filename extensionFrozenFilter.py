import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

import numpy as np
import cv2
import torch.optim as optim
from task1F import loadNetwork
from main import handleLoadingMNISTDataSet, test

# set the random seed
torch.manual_seed(2502)
torch.backends.cudnn.enabled = False

# global variables
n_epochs = 5
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



# method to train the network
def train(epoch, train_loader, network, optimizer, train_losses, train_counter):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            



# load the MNIST dataset
train_data, test_data = handleLoadingMNISTDataSet()



train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_data.dataset) for i in range(n_epochs + 1)]

# load the saved models
gabor_model = loadNetwork()
gabor_optimizer = optim.SGD(gabor_model.parameters(), lr=learning_rate, momentum=momentum)

laplacian_model = loadNetwork()
laplacian_optimizer = optim.SGD(laplacian_model.parameters(), lr=learning_rate, momentum=momentum)

gaussian_model = loadNetwork()
gaussian_optimizer = optim.SGD(gaussian_model.parameters(), lr=learning_rate, momentum=momentum)


# test the models with some input data
input_data = cv2.imread(r"D:/Users/mohit/MK/PRCV/Project5/recognition_using_deep_networks-master/files/handwrittenDigits/0/0.jpg", cv2.IMREAD_GRAYSCALE)
input_tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).float()

# replace the weights of the first conv layer with Gabor filters and freeze the layer
gabor_weights = gabor_filter_weights(in_channels, out_channels, kernel_size, sigma, lambd, psi, gamma)
gabor_model.conv1.weight.data = gabor_weights
gabor_model.conv1.weight.requires_grad = False

print("Gabor model training")
for epoch in range(1, n_epochs + 1):
    train(epoch, train_data, gabor_model, gabor_optimizer, train_losses, train_counter)
    test(gabor_model, test_data, test_losses)

gabor_output = gabor_model(input_tensor)
print("Gabor Filter Model:", gabor_output.argmax().item())

# replace the weights of the first conv layer with Laplacian filters and freeze the layer
laplacian_weights = laplacian_filter_weights(in_channels, out_channels)
laplacian_model.conv1.weight.data = laplacian_weights.float()
laplacian_model.conv1.weight.requires_grad = False

print("Laplacian model training")
for epoch in range(1, n_epochs + 1):
    train(epoch, train_data, laplacian_model, laplacian_optimizer, train_losses, train_counter)
    test(laplacian_model, test_data, test_losses)

laplacian_output = laplacian_model(input_tensor)
print("Laplacian Filter Model:", laplacian_output.argmax().item())

# replace the weights of the first conv layer with Gaussian filters and freeze the layer
gaussian_weights = gaussian_filter_weights(in_channels, out_channels, kernel_size, sigma)
gaussian_model.conv1.weight.data = gaussian_weights
gaussian_model.conv1.weight.requires_grad = False

print("Gaussian model training")
for epoch in range(1, n_epochs + 1):
    train(epoch, train_data, gaussian_model, gaussian_optimizer, train_losses, train_counter)
    test(gaussian_model, test_data, test_losses)

gaussian_output = gaussian_model(input_tensor)
print("Gaussian Filter Model:", gaussian_output.argmax().item())









