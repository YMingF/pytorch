#!/opt/anaconda3/envs/pytorch/bin/python

from torch.nn import MaxPool2d
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    "./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor()
)
dataloader = DataLoader(dataset, batch_size=64)


class maxPool(nn.Module):
    def __init__(self):
        super(maxPool, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        x = self.maxpool(input)
        return x


maxPool11 = maxPool()
writer = SummaryWriter("./maxPoolLogs")
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input", imgs, step)
    output = maxPool11(imgs)
    writer.add_images("output", output, step)
    step += 1
    break
writer.close()
