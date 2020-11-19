
import torch
from torchvision import transforms
from torchvision.utils import save_image
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
from PIL import Image
import os

SIGMAS = [1.0,2.0,3.0,4.0]
RESIZE = [224, 224]
FILES = ["test.png"]
NUM_CROPS = 32
NUM_REG = 1 #number of regression params. to predict
filename = FILES[0]

filename_noext=os.path.splitext(os.path.basename(filename))[0]

# Read image
#image = io.imread(fname=filename)
image = Image.open(filename)
image_tensor=transforms.functional.to_tensor(image).unsqueeze_(0)

tCROP = transforms.Compose([transforms.RandomCrop(size=(RESIZE[0],RESIZE[1])),])
tGAUSSIAN = [transforms.Compose([transforms.GaussianBlur(kernel_size=(7,7), sigma=SIGMAS[idx]),]) for idx in range(len(SIGMAS))]

x=[]
y=[]
for gidx in range(len(SIGMAS)):
    preproc_image=tGAUSSIAN[gidx](image_tensor)
    for cidx in range(NUM_CROPS):
        preproc_image=tCROP(preproc_image)
        save_image(preproc_image,filename_noext+"_blur_"+"rcrop_"+str(cidx+1)+"_sigma"+str(SIGMAS[gidx])+".png")
        x.append(preproc_image)
        y.append(torch.tensor(SIGMAS[gidx]))
x = torch.cat(x,dim=0)#torch.stack(x).float()
y = torch.stack(y)

torch.manual_seed(1)    # reproducible
x, y = Variable(x), Variable(y)
x.requires_grad=True

#net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
net = models.resnet18()
net.fc = torch.nn.Linear(512, NUM_REG)
#net = models.alexnet()
#net = models.vgg16()
#net.classifier=nn.Linear(512, num_classes)

# print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.MSELoss()  #torch.nn.BCEWithLogitsLoss() #torch.nn.MSELoss() 

my_images = []
#fig, ax = plt.subplots(figsize=(12,7))

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
