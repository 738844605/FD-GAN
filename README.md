# FD-GAN
## FD-GAN: Generative adversarial Networks with Fusion-discriminator for Single Image Dehazing(AAAI'20)
[PAPER](https://arxiv.org/abs/2001.06968)

[Yu Dong](https://github.com/WeilanAnnn).  [Yihao Liu](https://github.com/DoctorYy)

<center >
    <img src= "https://github.com/WeilanAnnn/FD-GAN/blob/master/facades/network.png"/>
</center>


In this paper, we propose a fully end-to-end algorithm FD-GAN for image dehazing. Moreover, we develop a novel Fusion-discriminator which can integrate the frequency information as additional priors and constraints into the dehazing network. Our method can generate more visually pleasing dehazed results with less color distortion. Extensive experimental results have demonstrated that our method performs favorably against several state-of-the-art methods on both synthetic datasets and real-world hazy images.

<center >
    <img src="https://github.com/WeilanAnnn/FD-GAN/blob/master/facades/RealImage.png" width="1200"/>
</center>

## Prerequisites
1. Ubuntu 18.04
2. Python 3.6
3. NVIDIA GPU + CUDA CuDNN (CUDA 8.0)

## Installation with low pytorch version
1. pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
2. pip install torchvision == 0.2
3. pip install scipy==1.0.0
4. pip install scikit-image,opencv-python,pillow

## Try to run demo.py with pytorch 1.10.0
Make following changes in demo.py
```
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
	name = k[7:] # remove 'module.'
	#
	name = name.replace('.norm.1.', '.norm1.')
	name = name.replace('.norm.2.', '.norm2.')
	name = name.replace('.conv.1.', '.conv1.')
	name = name.replace('.conv.2.', '.conv2.')
	#
	new_state_dict[name] = v
# load params
netG.load_state_dict(new_state_dict)
```

## Test using pre-trained model
```
python test.py
```
## Demo using pre-trained model
Since the proposed method uses hdf5 file to load the traning samples, the **generate_testsample.py** helps you to creat the testing or training sample yourself.

If your images are real:
```
python demo.py --valDataroot ./facades/'your_folder_name' --netG ./testmodel/netG_epoch_real.pth
```
If your images are synthetic:
```
python demo.py --valDataroot ./facades/'your_folder_name' --netG ./testmodel/netG_epoch_synthetic.pth
```
To obtain the best performance on synthetic and real-world datasets respectively, we provide two models from different  iterations in one  training procedure. In addition, please use netG.train() for testing since the batch for training is 1.

Pre-trained dehazing models can be downloaded at (put it in the folder '**test_model**'):
https://pan.baidu.com/s/10IgnZ0YiGsUxrgxoQQhsOg

## Metric
You can run the **PSNRSSIM.py** for quantitative results
```
python PSNRSSIM.py --gt_dir ./your_folder_name --result_dir ./your_folder_name
```

## Datasets
 You can download our synthetic test-data: RESIDE-SOTS and NTIRE dataset(strored in Hdf5 file and PNG) as following URL： 
https://pan.baidu.com/s/1oZwVX8FWFNzRaY_JyVB1pA

https://pan.baidu.com/s/1U6RjKF-UYXvBIHDt7SU0Ww

## How to read the Hdf5 file
Following are the sample python codes how to read the Hdf5 file:
```
import matplotlib.pyplot as plt
file_name=self.root+'/'+str(index)+'.h5'
f=h5py.File(file_name,'r')

gt=f['gt'][:]
haze=f['haze'][:]
plt.subplot(1,2,1), plt.title('gt')
plt.imshow(gt)
plt.subplot(1,2,2),plt.title('haze')
plt.imshow(haze)
plt.show()
```
## Citation


## Acknowledgments
Thank all co-authors so much!
