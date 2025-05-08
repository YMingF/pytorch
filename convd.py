#!/opt/anaconda3/envs/pytorch/bin/python
# 进行一些简单卷积操作的代码。
import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10(
    "./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


realTest = test()
writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, shapes = data
    output = realTest(imgs)
    print(output.shape)
    writer.add_images("input", imgs, step)
    # 将经过卷积后的图像，重新 reshape 成 3 个 channel，因为 tensorBoard只能展示 3 个 channel 的图像
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1
    break
