# Dhruv Kamalesh Kumar
# Yalala Mohit
# 04-04-2023

# imports
import torch
import task1F
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from main import plotLosses as plotLosses
import matplotlib.pyplot as plt


# hyper-parameters
learning_rate = 0.01
epochs = 300
train_batch_size = 16
test_batch_size = 16
reg_lambda = 0.01

# setting the seed
torch.manual_seed(2502)

# method to load the network and replace the last layer with a new layer
def loadNetwork():
    newNetwork = task1F.loadNetwork()
    # freezes the parameters for the whole network
    for param in newNetwork.parameters():
        param.requires_grad = False
    # replace the last layer with a new Linear layer with three nodes
    newNetwork.fc2 = torch.nn.Linear(50, 3)
    return newNetwork


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# DataLoader for the Greek data set
greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder("./files/greek_train/",
                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                               GreekTransform(),
                                                                               torchvision.transforms.Normalize(
                                                                                   (0.1307,), (0.3081,))])),
    batch_size=train_batch_size,
    shuffle=True)

greek_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder("./files/greek_test/",
                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                               GreekTransform(),
                                                                               torchvision.transforms.Normalize(
                                                                                   (0.1307,), (0.3081,))])),
    batch_size=test_batch_size,
    shuffle=True)

# printing the modified network
network = loadNetwork()
print(network)

# optimizer
optimizer = optim.SGD(network.fc2.parameters(), lr=learning_rate)

# method to train the network
def train(epoch, network, train_losses, train_counter):
    network.train()
    for batch_idx, (data, target) in enumerate(greek_train):
        optimizer.zero_grad()
        pred = network(data)
        loss = F.cross_entropy(pred, target)
        # add L2 regularization to the loss function
        regularization_loss = 0
        for param in network.parameters():
            regularization_loss += torch.sum(torch.square(param))
        # Regularized loss - to prevent overfit
        loss = loss + reg_lambda*regularization_loss
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("Epoch: {} \tLoss: {:.6f}".format(
                epoch, loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(greek_train.dataset)))
            torch.save(network.state_dict(), './results/model_greek.pth')

# method to test the network
def test(network, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in greek_test:
            output = network(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(greek_test.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(greek_test.dataset),
        100. * correct / len(greek_test.dataset)))

# method to get the label
def getLabel(index):
    if index == 0:
        return "Alpha"
    elif index == 1:
        return "Beta"
    elif index == 2:
        return "Gamma"

# training the network
network.train()
optimizer.zero_grad()

# loss params
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(greek_train.dataset) for i in range(epochs + 1)]

test(network, test_losses)
for epoch in range(1, epochs+1):
    train(epoch, network, train_losses, train_counter)
    test(network, test_losses)
torch.save(network.state_dict(), './results/model_greek.pth')
# plotting the losses
plotLosses(train_losses, test_losses, train_counter, test_counter)

examples = enumerate(greek_test)
batch_idx, (example_data, example_targets) = next(examples)

# predicting the output for the first 6 image in the test set
network.eval()
with torch.no_grad():
    output = network(example_data)
plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Predicted: {}".format(
        getLabel(output.data.max(1, keepdim=True)[1][i].item())))
    plt.xticks([])
    plt.yticks([])
plt.show()

# just to see how it performs on training data
network.eval()
examples2 = enumerate(greek_train)
batch_idx_train, (example_data_train, example_targets_train) = next(examples2)
with torch.no_grad():
    output_train = network(example_data_train)
plt.figure()
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data_train[i][0], cmap='gray', interpolation='none')
    plt.title("Predicted Train: {}".format(
        getLabel(output_train.data.max(1, keepdim=True)[1][i].item())))
    plt.xticks([])
    plt.yticks([])
plt.show()


