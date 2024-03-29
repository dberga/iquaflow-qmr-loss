import glob
import cv2
import kornia
import os

from torchvision.transforms import ColorJitter
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def generate_HR_LW_pair(img_array, crop_size=(256,256), scale=2, mode='training'):
    img = kornia.image_to_tensor(img_array.copy()).float()
    
    random_crop = kornia.augmentation.RandomCrop(crop_size,)
    random_center_crop = kornia.augmentation.CenterCrop(crop_size,)

    sigma = 0.5*scale
    kernel_size = int(sigma*3 + 4)
    
    if mode=='training':
        img_crop = random_crop(img)
    else:
        img_crop = random_center_crop(img)
        
    kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size,kernel_size), (sigma, sigma))
    blurred = kornia.filters.filter2D(img_crop, kernel_tensor[None])
    blurred_resize = kornia.geometry.rescale(blurred, 1/scale, 'bilinear')
    return img_crop[0], blurred_resize[0]
    
    
class DatasetHR_LR(Dataset):
    def __init__(self, mode='training',crop_size=(256,256), scale=2, apply_color_jitter=False, input_path="/scratch/SISR/datasets/dataset1/60cm/training/"):
        super(DatasetHR_LR, self).__init__()

        valid_fmt=['.jpg','.jpeg','.png','.tif']
        list_1=[]
        for f in os.listdir(input_path):
            fmt=os.path.splitext(f)[1]
            if fmt.lower() not in fmt:
                continue
            list_1.append(os.path.join(input_path,f))
        #list_1 = np.array(glob.glob(f"/scratch/SISR/datasets/dataset1/60cm/{mode}/*.png"))
        #list_2 = np.loadtxt(f"/scratch/SISR/datasets/dataset2/60cm/{mode}_HR_crops_r0.6_512.txt", dtype='str')
        #self.list_images = np.concatenate([list_1, list_2]) 
        self.list_images = np.array(list_1)
        self.mode=mode
        self.crop_size=crop_size
        self.scale=scale
        self.apply_color_jitter = apply_color_jitter
        self.colorjitter = ColorJitter(brightness=1, contrast=1, saturation=1, hue=0)

    def __getitem__(self, index):
        filename = self.list_images[index]
        img_array = cv2.imread(filename)
        
        
        if self.mode=='training' and (self.apply_color_jitter):
            img_array = self._apply_jitter(img_array)
            
        if self.mode=='trainig':
            if np.random.choice([0,1])==0:
                img_array = img_array[:,::-1,:]
            if np.random.choice([0,1])==0:
                img_array = img_array[::-1,:,:]
                

        img_HR, img_LR = generate_HR_LW_pair(img_array,self.crop_size, self.scale, self.mode)
            
        img_HR /= 255
        img_LR /= 255

        return img_LR, img_HR
    
    def _apply_jitter(self, img_array):
        transform = ColorJitter.get_params(
            self.colorjitter.brightness, self.colorjitter.contrast,
            self.colorjitter.saturation, self.colorjitter.hue)
        
        img_array = Image.fromarray(img_array)

        img_array = transform(img_array)
        
        return np.asarray(img_array)
    
    def __len__(self):
        return self.list_images.shape[0]
