# Perform standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import numpy as np
import pandas as pd
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import time

# Define device for XLA
device = xm.xla_device()
print(f"Running on XLA device: {device}")

# Load the MNIST dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)

# Use XLA-specific distributed sampler for data loading
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
)

train_loader = DataLoader(train_data, batch_size=10, sampler=train_sampler, num_workers=4, drop_last=True)
test_loader = DataLoader(test_data, batch_size=10, sampler=test_sampler, num_workers=4, drop_last=False)

# Define our convolutional neural network model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)  # Flatten the tensor
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

# Set random seed for reproducibility
torch.manual_seed(42)
model = ConvolutionalNetwork()

# Move the model to the XLA device
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * xm.xrt_world_size())

# Define the training model function
def train_fn(data_loader, model, optimizer, device, scheduler, epoch, num_steps):
    model.train()
    for bi, (X_train, y_train) in enumerate(data_loader):
        # Move data to XLA device
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        # Zero the gradients, backward pass, and optimizer step
        optimizer.zero_grad()
        loss.backward()

        # XLA-specific optimizer step
        xm.optimizer_step(optimizer, barrier=True)

        # Print interim results
        if (bi + 1) % 600 == 0:
            print(f'epoch: {epoch:2}  batch: {bi + 1:4} [{10 * (bi + 1):6}/60000]  loss: {loss.item():10.8f}')

def run(epochs=5):
    start_time = time.time()

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    best_accuracy = 0

    # Training loop
    for epoch in range(epochs):
        trn_corr = 0
        tst_corr = 0

        # Train model
        train_fn(train_loader, model, optimizer, device, scheduler=None, epoch=epoch, num_steps=len(train_loader))

        # Evaluate model
        model.eval()
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_val = model(X_test)
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test).sum()

        accuracy = tst_corr.item() * 100 / len(test_loader.dataset)
        print(f"Validation accuracy after epoch {epoch}: {accuracy:.2f}%")

        # Save model checkpoint
        if accuracy > best_accuracy:
            xm.save(model.state_dict(), "best_model.pth")
            best_accuracy = accuracy

    print(f"\nDuration: {time.time() - start_time:.0f} seconds")

# Define benchmarking function
def benchmark_model(model, test_loader, device):
    model.eval()
    total_time = 0
    processed_samples = 0

    start_time = time.time()

    with torch.no_grad():
        for b, (X_test, _) in enumerate(test_loader):
            X_test = X_test.to(device)

            # Time the inference of the batch
            start_batch = time.time()
            _ = model(X_test)  # Forward pass
            total_time += time.time() - start_batch
            processed_samples += X_test.size(0)

    total_time = time.time() - start_time
    throughput = processed_samples / total_time

    print(f'Total inference time: {total_time:.3f} seconds')
    print(f'Throughput: {throughput:.2f} samples/second')

    return total_time, throughput

# Execute the training and evaluation
run()

# Perform benchmarking after training
benchmark_model(model, test_loader, device)
