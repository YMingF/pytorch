#!/opt/anaconda3/envs/pytorch/bin/python
import torchvision

from torch.utils.data import DataLoader

# 测试数据集
test_data = torchvision.datasets.CIFAR10(
    root="/Users/alex/myProject/pytorch/data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
)

test_loader = DataLoader(
    test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False
)
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
writer.add("img")
