# Dhruv Kamalesh Kumar
# Yalala Mohit
# 04-04-2023

# Importing the required libraries
import task1F
import torch
import cv2
import matplotlib.pyplot as plt

from main import handleLoadingMNISTDataSet

# get the saved network
network = task1F.loadNetwork()

# task 2A - analyze the 1st layer of the network
print("Shape of 1st layer of the network - {}".format(network.conv1.weight.shape))

# getting the 1st filter and its shape
firstFilter = network.conv1.weight[0, 0]
print("Shape of 1st filter - {}".format(firstFilter.shape))
print("1st filter - {}".format(firstFilter))

# plotting the 1st layer of the network
with torch.no_grad():
    plt.figure()
    for i in range(len(network.conv1.weight)):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(network.conv1.weight[i, 0], interpolation='none')
        plt.title("Filter {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# task 2B - Showing the effect of the filters
# get one test image from MNIST test dataset
train_loader, test_loader = handleLoadingMNISTDataSet()
testImage = test_loader.dataset[0][0]
print("Shape of test image - {}".format(testImage.shape))

