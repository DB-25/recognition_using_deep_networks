import torch
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Load the pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)

def analyze_layer(layer_num):
    """
    Analyzes the specified layer of the VGG16 network.
    """
    layer = vgg16.features[layer_num]
    print(f"Shape of layer {layer_num + 1} of the network - {layer.weight.shape}")
    
    # get the first filter and its shape
    first_filter = layer.weight[0, 0]
    print(f"Shape of first filter - {first_filter.shape}")
    print(f"First filter - {first_filter}")
    
    # plot the layer's filters
    with torch.no_grad():
        plt.figure()
        for i in range(len(layer.weight)):
            plt.subplot(8, 8, i + 1)
            plt.tight_layout()
            plt.imshow(layer.weight[i, 0].cpu().numpy(), interpolation='none')
            plt.title(f"Filter {i}")
            plt.xticks([])
            plt.yticks([])
        plt.show()
        
def filter_image(layer_num, img_path):
    """
    Applies the specified layer's filters to the input image.
    """
    # Load an example image and preprocess it
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    vgg16.to(device)
    
    # apply the filters to the input image
    with torch.no_grad():
        plt.figure()
        for i in range(len(vgg16.features[layer_num].weight)*2):
            if i >=64:
                break
            plt.subplot(8, 8, i+1)
            plt.tight_layout()
            if i % 2 == 0:
                # display filter from network
                plt.imshow(vgg16.features[layer_num].weight[i//2, 0].cpu().numpy(), cmap="gray", interpolation='none')
                plt.title(f"Filter {i//2 + 1}")
            else:
                # apply filter to input image
                output = vgg16.features[:layer_num+1](input_batch)
                filtered_image = cv2.filter2D(output[0, i//2].cpu().numpy(), -1, vgg16.features[layer_num].weight[i//2, 0].cpu().numpy())
                plt.imshow(filtered_image, cmap="gray", interpolation='none')
                plt.title(f"Filtered Image {i//2 + 1}")
            plt.xticks([])
            plt.yticks([])
        plt.show()

# task 2A - analyze the first 2 layers of the network
analyze_layer(0)
analyze_layer(2)

# task 2B - apply the filters to an input image
filter_image(0, "car1.jpg")
filter_image(2, "car1.jpg")
filter_image(0, "dog1.jpg")
filter_image(2, "dog1.jpg")