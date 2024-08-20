import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time

# Setting the device to CUDA if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform and then loading the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Loading the pre-trained ResNet50 model and then move it to the CUDA device (GPU) or CPU
model = models.resnet50(pretrained=True).to(device)

# Warm-up and initializing the device
def warmup(model, loader, device):
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _ = model(data)
    print("Warmed up the model on the device.")

# Benchmark inference with the PyTorch compiler
def benchmark_inference_pytorch(model, loader, device, iterations=100):
    model.eval()
    total_time = 0
    processed_samples = 0
    
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= iterations:
                break
            data = data.to(device)
            start_time = time.time()
            _ = model(data)
            total_time += time.time() - start_time
            processed_samples += data.size(0)
    
    throughput = processed_samples / total_time
    print(f"Total inference time->(PyTorch): {total_time:.3f} seconds")
    print(f"Throughput->(PyTorch): {throughput:.2f} samples/second")
    return total_time, throughput


warmup(model, train_loader, device)
benchmark_inference_pytorch(model, train_loader, device)
