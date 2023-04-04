# Dhruv Kamalesh Kumar
# Yalala Mohit
# 03-30-2023

# Importing the required libraries
from main import NeuralNetwork
import torch


def loadNetwork():
    # read the network architecture from the file
    # create a network with the architecture
    network = NeuralNetwork()

    network.load_state_dict(torch.load('./results/model.pth'))
    # print the network architecture
    print(network)
    return network
