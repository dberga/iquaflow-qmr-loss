# Eurecat's Development repo

We will use this repo for research and development.

# Contents

(LPR-IQA) Learning Parameter Regression for Image Quality Assessment
(SRGAN-PyTorch) Pytorch implementation of SRGAN algorithm

## Useful Links

* [Regression with Pytorch](https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379)
* [Blurring with Scikit-Image](https://datacarpentry.org/image-processing/06-blurring/)
* [Dilated Network](https://medium.com/@vaibhaw.vipul/building-a-dilated-convnet-in-pytorch-f7c1496d9bf5)
* [Pytorch Transforms in single image](https://discuss.pytorch.org/t/applying-transforms-to-a-single-image/56254)
* [Autograd loss problem](https://stackoverflow.com/questions/64513183/pytorch-not-updating-weights-when-using-autograd-in-loss-function)
* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/Lornatang/SRGAN-PyTorch)
## Install NVIDIA driver (440)
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get install nvidia-driver-440
```

## Install Torch Environment and dependencies (CUDA 10.2)
```
conda create -n satellogic python=3.7 
conda install -n satellogic pytorch torchvision python-graphviz matplotlib glob2 xlrd pillow numpy scipy scikit-image intel-openmp mkl cudatoolkit=10.2 -c pytorch
pip install git+https://github.com/bodleian/image-processing.git
```
