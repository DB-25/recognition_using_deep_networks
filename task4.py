# Dhruv Kamalesh Kumar
# Yalala Mohit
# 04-04-2023

import warnings
warnings.filterwarnings("ignore")

# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# calculate train time, writing train data to files etc.
import time
import pandas as pd
import json
from IPython.display import clear_output

# TensorBoard support
from torch.utils.tensorboard import SummaryWriter 

# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
from collections import namedtuple
from itertools import product

# use scikitplot to plot the confusion matrix
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import scikitplot as skplt

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)     # On by default, leave it here for clarity

# check PyTorch versions
print(torch.__version__)
print(torchvision.__version__)

# Use standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)

test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)


# Build the neural network, expand on top of nn.Module
class Network(nn.Module):
    def __init__(self, conv_channels, conv_kernel_size, pool_kernel_size, pool_stride, dropout_rate, hidden_layers, activation):
        super().__init__()

        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels[0], kernel_size=conv_kernel_size)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=conv_channels[1], kernel_size=conv_kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.dropout = nn.Dropout(p=dropout_rate)

        # calculate the in_features for the first linear layer
        in_features = self._calculate_in_features()

        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_layers[0])
        self.fc2 = nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1])

        
        self.activation = activation
        self.out = nn.Linear(in_features=hidden_layers[1], out_features=10)

    # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = self.activation(t)
        t = self.pool(t)

        # conv 2
        t = self.conv2(t)
        t = self.activation(t)
        t = self.pool(t)

        # fc1
        t = t.reshape(t.size(0), -1)
        t = self.fc1(t)
        t = self.activation(t)
        t = self.dropout(t)

        # fc2
        t = self.fc2(t)
        t = self.activation(t)
        t = self.dropout(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.
        return t

    def _calculate_in_features(self):
        # Create an example input and pass it through the network to get the output size
        x = torch.randn(1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x.flatten().shape[0]
        



def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()


# Read in the hyper-parameters and return a Run namedtuple containing all the 
# combinations of hyper-parameters
class RunBuilder():
  @staticmethod
  def get_runs(params):

    Run = namedtuple('Run', params.keys())

    runs = []
    for v in product(*params.values()):
      runs.append(Run(*v))
    
    return runs

# Helper class, help track loss, accuracy, epoch time, run time, 
# hyper-parameters etc. Also record to TensorBoard and write into csv, json
class RunManager():
  def __init__(self):

    # tracking every epoch count, loss, accuracy, time
    self.epoch_count = 0
    self.epoch_loss = 0
    self.epoch_num_correct = 0
    self.epoch_start_time = None

    # tracking every run count, run data, hyper-params used, time
    self.run_params = None
    self.run_count = 0
    self.run_data = []
    self.run_start_time = None

    # record model, loader and TensorBoard 
    self.network = None
    self.loader = None
    self.tb = None

  # record the count, hyper-param, model, loader of each run
  # record sample images and network graph to TensorBoard  
  def begin_run(self, run, network, loader):

    self.run_start_time = time.time()

    self.run_params = run
    self.run_count += 1

    self.network = network
    self.loader = loader
    self.tb = SummaryWriter(comment=f'-{self.run_count}')

    images, labels = next(iter(self.loader))
    grid = torchvision.utils.make_grid(images)

    self.tb.add_image('images', grid)
    self.tb.add_graph(self.network, images)

  # when run ends, close TensorBoard, zero epoch count
  def end_run(self, network, train_set, test_set, batch_size):
    self.tb.close()
    self.epoch_count = 0
    # bigger batch size since we only do FP
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size)
    train_preds = get_all_preds(network, prediction_loader)
    test_prediction_loader = torch.utils.data.DataLoader(test_set, batch_size)
    test_preds = get_all_preds(network, test_prediction_loader)
    train_cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
    test_cm = confusion_matrix(test_set.targets, test_preds.argmax(dim=1))
    print("Train Confusion Matrix = ",train_cm)
    print("Test Confusion Matrix = ",test_cm)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    plt.subplots_adjust(bottom=0.2)
    skplt.metrics.plot_confusion_matrix(train_set.targets, train_preds.argmax(dim=1), normalize=True, ax=ax1)
    skplt.metrics.plot_confusion_matrix(test_set.targets, test_preds.argmax(dim=1), normalize=True, ax=ax2)

    plt.figtext(0.5, 0.05,
                f"Train accuracy = {_get_accuracy(train_preds, train_set.targets)}, Test accuracy = {_get_accuracy(test_preds, test_set.targets)}",
                ha="center", 
                fontsize=6, 
                transform=plt.gcf().transFigure)

    plt.savefig(f"task_4_results/Confusion-matrix - run {self.run_count}.png")


  # zero epoch count, loss, accuracy, 
  def begin_epoch(self):
    self.epoch_start_time = time.time()

    self.epoch_count += 1
    self.epoch_loss = 0
    self.epoch_num_correct = 0

  # 
  def end_epoch(self):
    # calculate epoch duration and run duration(accumulate)
    epoch_duration = time.time() - self.epoch_start_time
    run_duration = time.time() - self.run_start_time

    # record epoch loss and accuracy
    loss = self.epoch_loss / len(self.loader.dataset)
    accuracy = self.epoch_num_correct / len(self.loader.dataset)

    # Record epoch loss and accuracy to TensorBoard 
    self.tb.add_scalar('Loss', loss, self.epoch_count)
    self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

    # Record params to TensorBoard
    for name, param in self.network.named_parameters():
      self.tb.add_histogram(name, param, self.epoch_count)
      self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
    
    # Write into 'results' (OrderedDict) for all run related data
    results = OrderedDict()
    results["run"] = self.run_count
    results["epoch"] = self.epoch_count
    results["loss"] = loss
    results["accuracy"] = accuracy
    results["epoch duration"] = epoch_duration
    results["run duration"] = run_duration

    # Record hyper-params into 'results'
    for k,v in self.run_params._asdict().items(): results[k] = v
    self.run_data.append(results)
    df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')

    # display epoch information and show progress
    clear_output(wait=True)
    print(df)

  # accumulate loss of batch into entire epoch loss
  def track_loss(self, loss):
    # multiply batch size so variety of batch sizes can be compared
    self.epoch_loss += loss.item() * self.loader.batch_size

  # accumulate number of corrects of batch into entire epoch num_correct
  def track_num_correct(self, preds, labels):
    self.epoch_num_correct += self._get_num_correct(preds, labels)

  @torch.no_grad()
  def _get_num_correct(self, preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
  
  # save end results of all runs into csv, json for further a
  def save(self, fileName):

    pd.DataFrame.from_dict(
        self.run_data, 
        orient = 'columns',
    ).to_csv(f'{fileName}.csv')

    with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
      json.dump(self.run_data, f, ensure_ascii=False, indent=4)

# helper function to calculate all predictions of train set
def get_all_preds(model, loader):
  all_preds = torch.tensor([])
  for batch in loader:
    images, labels = batch

    preds = model(images)
    all_preds = torch.cat(
        (all_preds, preds),
        dim = 0
    )
  return all_preds

@torch.no_grad()
def _get_accuracy(preds, labels):
  return (preds.argmax(dim=1).eq(labels).sum().item())/len(labels)

# put all hyper params into a OrderedDict, easily expandable
params = OrderedDict(
    lr = [.01, .001],
    batch_size = [100, 1000],
    shuffle = [False],
    epochs = [5,10],
    conv_channels = [[6,12], [12,12], [8,16]], 
    conv_kernel_size = [3,5,88],
    pool_kernel_size = [3,6,9], 
    pool_stride = [1, 2, 3],
    dropout_rate = [0.1, 0.5, 0.7], 
    hidden_layers = [[120, 60], [512,256], [1024,512]], 
    activation = [nn.ReLU(), nn.ReLU6(), nn.Tanh()],
)

m = RunManager()

# get all runs from params using RunBuilder class
for run in RunBuilder.get_runs(params):

    # if params changes, following line of code should reflect the changes too
    # network = Network()
    network = Network(run.conv_channels, run.conv_kernel_size, run.pool_kernel_size, run.pool_stride, run.dropout_rate, run.hidden_layers, run.activation)
    
    print(network)
    loader = torch.utils.data.DataLoader(train_set, batch_size = run.batch_size, shuffle = run.shuffle)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(run.epochs):
      
      m.begin_epoch()
      for batch in loader:
        
        images = batch[0]
        labels = batch[1]
        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        m.track_loss(loss)
        m.track_num_correct(preds, labels)

      m.end_epoch()
    m.end_run(network, train_set, test_set, batch_size=1000)

# when all runs are done, save results to files
m.save('results')



