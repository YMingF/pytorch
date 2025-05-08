#!/opt/anaconda3/envs/pytorch/bin/python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

writer = SummaryWriter("logs")
img = Image.open(
    "/Users/alex/myProject/pytorch/dataset/train/ants/5650366_e22b7e1065.jpg"
)
print(type(img))
import numpy as np

img_array = np.array(img)

writer.add_image("test", img_array, 2, dataformats="HWC")
writer.close()
