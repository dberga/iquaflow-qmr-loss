# Eurecat's Development repo

We will use this repo for research and development.

# Contents

(LPR-IQA) Learning Parameter Regression for Image Quality Assessment

# Requirements

*  Test 1: Learning single parameter (sigma) from image blurring
> Note: Assuming to use a pretrained network for classification (we remove the classifier and keep the feature extractor), we'd need to create a small set in order to train the (new) regression Linear layer.

## Useful Links

* [Regression with Pytorch](https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379)
* [Blurring with Scikit-Image](https://datacarpentry.org/image-processing/06-blurring/)

## Install NVIDIA driver (440)
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get install nvidia-driver-440
```

## Install Torch Environment and dependencies (CUDA 10.2)
```
conda create -n satellogic python=3.5 
conda install -n satellogic pytorch torchvision python-graphviz matplotlib glob2 xlrd pillow numpy scipy scikit-image intel-openmp mkl cudatoolkit=10.2 -c pytorch
pip install git+https://github.com/bodleian/image-processing.git
```