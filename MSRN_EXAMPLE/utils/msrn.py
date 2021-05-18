import kornia
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

class WindowsDataset_SR(data.Dataset):
    def __init__(self, nimg, wind_size=512, stride=480, scale=2):
        
        self.nimg = nimg
        H, W, C = nimg.shape
        self.H = H
        self.W = W
        self.C = C
        self.scale = scale
        
        self.H_out = H*scale
        self.W_out = W*scale

        self.stride = stride
        self.wind_size = wind_size
                
        # get all locations
        self.coordinates_input = []
        for j in range(0, self.H, self.stride):
            for i in range(0, self.W, self.stride):
                y0=j
                y1=min(j+self.wind_size, self.H)
                x0=i
                x1=min(i+self.wind_size, self.W)
                
                self.coordinates_input.append((y0,y1,x0,x1))
            
    def __len__(self):
        return len(self.coordinates_input)

    def __getitem__(self, index):
        y0,y1,x0,x1=self.coordinates_input[index]
        w = x1-x0
        h = y1-y0
        crop = np.zeros((self.wind_size,self.wind_size,3))

        crop[:h, :w] = self.nimg[y0:y1, x0:x1]

        # normalization
        crop = crop
        x_in = kornia.image_to_tensor(crop).float()
        
        sample = {
            'x_in':x_in,
            'coordinates': np.array(self.coordinates_input[index]),
            'w': w,
            'h': h
        }
        return sample


def inference_model(model, nimg, wind_size=512, stride=480, scale=2, 
                    batch_size=1, data_parallel=False, padding=5, manager=None):
    """
    Run sliding window on data using the sisr model.
    
    args:
        model
        data (H,W,C) in BGR format normalized between 0-1 (float)
        wind_size
        stride
    returns:
        super resolved image xscale. Numpy array (H,W,C) BGR 0-1 (float)
    """
    
    # get device
    for p in model.parameters():
        device = p.get_device()
        break
    if device ==-1:
        device = 'cpu'
    elif data_parallel:
        print("Using multiple GPU!")
        model = nn.DataParallel(model).cuda()
           
    H,W,C=nimg.shape
    
        
    # init dataset 
    dataset = WindowsDataset_SR(nimg, wind_size, stride, scale)
    
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    
    output = np.zeros((dataset.H_out, dataset.W_out, C)).astype(np.float)
    counts = np.zeros((dataset.H_out, dataset.W_out, C)).astype(np.float)

    psteps = tqdm(total=len(dataloader), desc='\tSI-AI inference', position=0)

    if manager is not None:
        psteps = manager.counter(total=len(dataloader), desc='\tSI-AI inference', unit='steps')

    for sample in dataloader:

        if not data_parallel:
            x_in = sample['x_in'].to(device)
        else:
            x_in = sample['x_in'].cuda()

        # add 5 pixel padding to avoid border effect
        x_in = F.pad(input=x_in, pad=(padding, padding, padding, padding), mode='reflect')

        pred_sate = model(x_in)
        pred_sate = pred_sate.detach().data.cpu().numpy()
        pred_sate = pred_sate[:,:,
                              scale*padding:-scale*padding,
                              scale*padding:-scale*padding]
        for ii in np.arange(pred_sate.shape[0]):
            pred_sample = pred_sate[ii].transpose((1,2,0))

            pred_sample = (np.clip((pred_sample), 0, 1))
            y0,y1,x0,x1 = sample['coordinates'][ii]
            h = sample['h'][ii].item()
            w = sample['w'][ii].item()

            Y0=y0*scale
            Y1=y1*scale
            X0=x0*scale
            X1=x1*scale          

            hh,ww,cc = output[Y0:Y1, X0:X1].shape

            if (hh<scale*wind_size) or (ww<scale*wind_size):
                Y1=Y0+hh
                X1=X0+ww
                pred_sample = pred_sample[:hh, :ww]

            output[Y0:Y1, X0:X1]+=pred_sample[...]
            counts[Y0:Y1, X0:X1,:]+=1
            psteps.update()
            
    res = np.divide(output, counts)
    return res




def load_msrn_model(weights_path=None, cuda='0'):
    """
    Load MSRN traied model on specific GPU for inference
    
    args:
        path to weights (default model_epoch_101.pth located in Nas) x2 scale
        cuda '0' or set to None if yoy want CPU usage
    
    return:
        pytorch MSRN
    """
    if cuda is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]=cuda
    model = MSRN_Upscale(n_scale=2)
    
    if weights_path is None:
        weights_path="model.pth"
        
    if not torch.cuda.is_available():
        weights = torch.load(weights_path,  map_location=torch.device('cpu'))
    else:
        weights = torch.load(weights_path)

    model.load_state_dict(weights)
    model.eval()
    
    if cuda is not None:
        model.cuda()
    
    print("Loaded MSRN ", weights_path)
    return model


# residual module
class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()
        self.n_feats = n_feats
        self.conv3_1 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=5, stride=1, padding=2)
        self.conv3_2 = nn.Conv2d(2*self.n_feats, 2*self.n_feats, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(2*self.n_feats, 2*self.n_feats, kernel_size=5, stride=1, padding=2)
        self.conv1_3 = nn.Conv2d(4*self.n_feats, self.n_feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x_input = x.clone()
        
        x3_1 = self.relu(self.conv3_1(x))
        x5_1 = self.relu(self.conv5_1(x))
        x1 = torch.cat([x3_1, x5_1], 1)
        
        x3_2 = self.relu(self.conv3_2(x1))
        x5_2 = self.relu(self.conv5_2(x1))
        x2 = torch.cat([x3_2, x5_2], 1)
        
        x_final = self.conv1_3(x2)
        return x_final + x_input
    
# Full structure
class MSRN_Upscale(nn.Module):
    def __init__(self, n_input_channels=3, n_blocks=8, n_feats=64, n_scale=4):
        super(MSRN_Upscale, self).__init__()
        
        self.n_blocks = n_blocks
        self.n_feats = n_feats
        self.n_scale = n_scale
        self.n_input_channels = n_input_channels
        
        # input
        self.conv_input = nn.Conv2d(self.n_input_channels, self.n_feats, kernel_size=3, stride=1, padding=1)
        
        # body
        conv_blocks = []
        for i in range(self.n_blocks):
            conv_blocks.append(MSRB(self.n_feats))
        self.conv_blocks = nn.Sequential(*conv_blocks)       
        
        self.bottle_neck = nn.Conv2d((self.n_blocks+1)* self.n_feats, self.n_feats, kernel_size=1, stride=1, padding=0)
              
        # tail
        self.conv_up = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1, bias=1)
        self.pixel_shuffle = nn.Upsample(scale_factor=self.n_scale, mode='nearest')
        self.conv_output = nn.Conv2d(self.n_feats, n_input_channels, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    
    def _init_pixel_shuffle(self):
        kernel = ICNR(self.conv_up.weight, scale_factor=self.n_scale)
        self.conv_up.weight.data.copy_(kernel)       
        
    def forward(self, x):
        x_input = x.clone()
        
        features=[]

        # M0
        x = self.conv_input(x)
        features.append(x)
        
        # body
        for i in range(self.n_blocks):
            x = self.conv_blocks[i](x)
            features.append(x)
            
        x = torch.cat(features, 1)
        
        x = self.bottle_neck(x)

        x = self.conv_up(x)
        x = self.pixel_shuffle(x)
        x = self.conv_output(x)
        
        return x