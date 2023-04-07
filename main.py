# Dhruv Kamalesh Kumar
# Yalala Mohit
# 03-30-2023

# Importing the required libraries
import sys
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import task1F
import task1G

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# global variables
n_epochs = 50
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the MNIST dataset
def handleLoadingMNISTDataSet():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    # print size of training and test data
    print("Size of training data: ", len(train_loader.dataset))
    print("Size of test data: ", len(test_loader.dataset))
    return train_loader, test_loader


# Task 1A
def task1A(train_data, train_targets):
    # using pyplot, plot the first 6 images in the training set in a subplot
    # with 2 rows and 3 columns use the title of the subplot to show the label of the image
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(train_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(train_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return


# Task 1C - define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # define the layers of the network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


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
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


# method to test the network
def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# method to plot the training and test losses
def plotLosses(train_losses, test_losses, train_counter, test_counter):
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# main function
def main(argv):
    # handle any command line arguments in argv

    # set the random seed - task 1B
    torch.manual_seed(2502)
    # torch.backends.cudnn.enabled = False

    # load the MNIST dataset
    train_data, test_data = handleLoadingMNISTDataSet()

    # Task 1A
    examples = enumerate(test_data)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    task1A(example_data, example_targets)

    # Task 1D
    # initialize the neural network and the optimizer
    network = NeuralNetwork().to(device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    optimizer.zero_grad()

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_data.dataset) for i in range(n_epochs + 1)]

    print("Do you want to train the network? (y/n)")
    answer = input()
    if answer == "y":
        # train the network
        test(network, test_data, test_losses)
        for epoch in range(1, n_epochs + 1):
            train(epoch, train_data, network, optimizer, train_losses, train_counter)
            test(network, test_data, test_losses)
        # Task 1E
        # save the model to a file
        torch.save(network.state_dict(), './results/model.pth')
        # plot the training and test losses
        plotLosses(train_losses, test_losses, train_counter, test_counter)
    else:
        # load the model from a file
        network = task1F.loadNetwork()
        network.to(device)
        network.eval()
        # test the model
        test(network, test_data, test_losses)
        # print predictions for 9 images in the test set
        with torch.no_grad():
            output = network(example_data.to(device))
        plt.figure()
        example_data = example_data.to('cpu')
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
            plt.xticks([])
            plt.yticks([])
        plt.show()

    # Task 1G
    task1G.testNetworkOnHandWrittenDigits(network)
    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
