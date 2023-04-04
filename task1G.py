# Dhruv Kamalesh Kumar
# Yalala Mohit
# 03-30-2023

# imports
import torch
import torchvision
import matplotlib.pyplot as plt

# function to load the test images from directory
def loadTestImages():
    # load the test images from the directory
    # convert images to grayscale
    # return testImageLoader
    testImageLoader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder('./files/handwrittenDigits/', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.invert(x)), torchvision.transforms.Grayscale(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=10, shuffle=True)
    return testImageLoader

# function to test the network on handwritten digits
def testNetworkOnHandWrittenDigits(network):
    # load the test data
    handwrittenDigits = loadTestImages()
    data = enumerate(handwrittenDigits)
    batch_idx, (testDataset, _) = next(data)
    print("Size of test data: ", len(testDataset))
    print(testDataset.shape)
    network.eval()
    # for each image in the test data
    # get the prediction from the network
    with torch.no_grad():
        predictions = network(testDataset)

    # let's plot the images with respective predictions
    plt.figure()
    for i in range(len(testDataset)):
        plt.subplot(4, 3, i + 1)
        plt.tight_layout()
        plt.imshow(testDataset[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(predictions.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
