"""
This code sets up an XLA (Accelerated Linear Algebra) device, loads the CIFAR-10 dataset, and benchmarks the inference performance of a pre-trained ResNet50 model on the XLA device.

The `warmup` function is used to warm up the model on the XLA device before running the benchmark. The `benchmark_inference_openxla` function measures the total inference time and throughput of the model on the XLA device.

The code demonstrates how to use PyTorch's XLA support to leverage the performance of a GPU for deep learning tasks.
"""
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import os

# Setting up XLA device
device = xm.xla_device()

# transform and then loading the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Loading the pre-trained ResNet50 model and then move it to the XLA device (GPU)
model = models.resnet50(pretrained=True).to(device)

# Warm-up and initializing the device
def warmup(model, loader, device):
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _ = model(data)
    print("Warmed up the model on the XLA device.")

# Benchmark inference with OpenXLA
def benchmark_inference_openxla(model, loader, device, iterations=100):
    model.eval()
    total_time = 0
    processed_samples = 0
    
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= iterations:
                break
            data = data.to(device)
            xm.mark_step()  
            start_time = time.time()
            _ = model(data)
            xm.mark_step()  
            total_time += time.time() - start_time
            processed_samples += data.size(0)
    
    throughput = processed_samples / total_time
    print(f"Total inference time->(OpenXLA): {total_time:.3f} seconds")
    print(f"Throughput->(OpenXLA): {throughput:.2f} samples/second")
    return total_time, throughput


warmup(model, train_loader, device)
benchmark_inference_openxla(model, train_loader, device)
