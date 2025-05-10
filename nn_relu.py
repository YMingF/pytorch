#!/opt/anaconda3/envs/pytorch/bin/python
import torch
from torch.nn import ReLU, Sigmoid
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(
    "./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor()
)

dataLoader = DataLoader(dataset, batch_size=64)


class hha(nn.Module):
    def __init__(self):
        super(hha, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        x = self.sigmoid(input)
        return x


writer = SummaryWriter("./sigmoidLogs")
real = hha()
steps = 0
for data in dataLoader:
    imgs, target = data
    writer.add_images("input", imgs, steps)
    output = real(imgs)

    writer.add_images("output", output, steps)
    steps += 1
    break

writer.close()
