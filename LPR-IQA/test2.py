
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.models as models
import matplotlib.pyplot as plt
#matplotlib inline
import sys
import numpy as np
import imageio
from skimage import io, filters, transform
import os

SIGMAS = [1.0,2.0,3.0,4.0]
RESIZE = [224, 224]
FILES = ["test.png"]
NUM_REG = 1 #number of regression params. to predict
filename = FILES[0]

filename_noext=os.path.splitext(os.path.basename(filename))[0]

# Read image
image = io.imread(fname=filename)


#x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
#y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

#preprocess (resize and blur)

#single image
#image = transform.resize(image,(224, 224,image.shape[2]))
#image = filters.gaussian(image, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

#alternative
#torchvision.transforms.functional.resize(img: torch.Tensor, size: List[int], interpolation: int = 2)
#torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
x=[]
y=[]
for idx in range(len(SIGMAS)):
    preproc_image = image
    preproc_image=transform.resize(preproc_image,(RESIZE[0], RESIZE[1],image.shape[2]))
    preproc_image=filters.gaussian(preproc_image, sigma=(SIGMAS[idx], SIGMAS[idx]), truncate=3.5, multichannel=True)
    io.imsave(filename_noext+"_blur_sigma"+str(SIGMAS[idx])+".png",image)
    x.append(torch.tensor(preproc_image))
    y.append(torch.tensor(SIGMAS[idx]))
x = torch.stack(x).float()
y = torch.stack(y)

x = x.view(x.shape[0],x.shape[3],x.shape[1],x.shape[2])

#save preprocessed image and create tensor
# torch can only train on Variable, so convert them to Variable
torch.manual_seed(1)    # reproducible
x, y = Variable(x), Variable(y)

#net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
net = models.resnet18()
net.fc = torch.nn.Linear(512, NUM_REG)
#net = models.alexnet()
#net = models.vgg16()
#net.classifier=nn.Linear(512, num_classes)

# print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.BCEWithLogitsLoss() #torch.nn.MSELoss()  # this is for regression mean squared loss

my_images = []
fig, ax = plt.subplots(figsize=(12,7))

print("Target=")
print(y)

# train the network
for t in range(200): #epoch
    #for b in range(len(x)): #batch (bs=1 image per batch)

    prediction = net(x)     # input x and predict based on x, [b,:,:,:]
    loss = loss_func(prediction.squeeze(), y)     # must be (1. nn output, 2. target)
    

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    print('Step = %d' % t+' Loss = %.4f' % loss.data.numpy()) #' Batch = %d' % b + 
    print("Prediction:")
    print(prediction.squeeze())

