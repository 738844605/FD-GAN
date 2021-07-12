import argparse

import PIL.Image as Image
import scipy.misc
import skimage.io as sio
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import cv2
import models.dehaze1113  as net
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#===== load model =====
print('===> Loading model')

netG = net.FDGAN()

state_dict = torch.load('./test_model/netG_epoch_synthetic.pth')
#torch.load('./test_model/netG_epoch_real.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
netG.load_state_dict(new_state_dict)
if torch.cuda.device_count() > 1:
    netG = nn.DataParallel(netG).cuda()


#===== Load input image =====
transform = transforms.Compose([
    transforms.ToTensor()
    ]
)
I_HAZE = 'IHAZE/hazy1'
O_HAZE='OHAZE/hazy1'

torch.backends.cudnn.benchmark = True
origin_dir = 'SOTSI/hazy'
output_dir = 'result/'
img_name_list = os.listdir(origin_dir)
img_sum = len(img_name_list)
print(img_sum)
time_sum = 0
for i in img_name_list:
    img_name_path = os.path.join(origin_dir, i)
    
    (imageName, extension) = os.path.splitext(i)
    print(imageName)
    #img = cv2.imread(img_name_path)
    img = Image.open(img_name_path).convert('RGB')
    imgIn = transform(img).unsqueeze_(0)
    varIn = Variable(imgIn)
    start = cv2.getTickCount()
    prediction = netG(varIn)
    prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))
    end  = cv2.getTickCount()
    time_sum+=(end - start) / cv2.getTickFrequency()
    save_path = os.path.join(output_dir+imageName+'.png')
    
    scipy.misc.toimage(prediction).save(save_path)
print(time_sum, img_sum)
print(time_sum/img_sum)
