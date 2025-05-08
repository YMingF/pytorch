#!/opt/anaconda3/envs/pytorch/bin/python
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "dataset/train/ants/6240329_72c01e663e.jpg"
img = Image.open(img_path)
trans_toTensor = transforms.ToTensor()
# img_tensor = transform(img)
writer = SummaryWriter("logs")
# writer.add_image("ToTensor", img_tensor)  # 就像把实验的一些图片贴到笔记本上。

trans_size = transforms.Resize((512, 512))
img_resize = trans_size(img)
img_resize_toTensor = trans_toTensor(img_resize)
writer.add_image("imgResize", img_resize_toTensor, 0)
print(img_resize)
# writer.close()

# trans_resize_2 = transforms.Resize((512, 512))
# trans_compose = transforms.Compose([trans_resize_2, transform])
# img_resize_2 = trans_compose(img)
trans_random = transforms.RandomCrop((20, 20))
trans_compose2 = transforms.Compose([trans_random, trans_toTensor])
img_random2 = trans_compose2(img)
writer.add_image("trans_compose5", img_random2, 0)
