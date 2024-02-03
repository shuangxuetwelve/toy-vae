import torch
import torchvision
from torchinfo import summary
from vae import VAE

# dataset[index] is a tuple (Tensor(3, 32, 32), int)
dataset = torchvision.datasets.CIFAR10('./data', download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

model = VAE(3)
summary(model)
