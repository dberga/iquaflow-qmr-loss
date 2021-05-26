import argparse

import cv2
import glob
import os
import sys
import time

import kornia
import numpy as np

from utils.msrn import load_msrn_model, inference_model

def generate_low_resolution_image(img_array, scale=2):
    img = kornia.image_to_tensor(img_array.copy()).float()[None]
    
    sigma = 0.5*(1/scale)
    kernel_size = int(sigma*3 + 4)
    if kernel_size%2==0:
        kernel_size=+1
            
    kernel_tensor = kornia.filters.get_gaussian_kernel2d((kernel_size,kernel_size), (sigma, sigma))
    blurred = kornia.filter2D(img, kernel_tensor[None])
    blurred_resize = kornia.geometry.rescale(blurred, scale, 'bilinear')
    print(blurred_resize.size())
    blurred_resize = blurred_resize.numpy()[0].transpose(1,2,0).astype(np.uint8)
    return blurred_resize



def process_file(model, path_out, compress=True, target_res=0.7, source_res=0.3,
                wind_size=512, stride=480, batch_size=1, padding=5, manager=None, filename=None):
    
    # image at source_res
    nimg = cv2.imread(filename)
    fmt_in = os.path.basename(filename).split('.')[1]
    fmt_out = "png" # png (or) same as fmt_in?

    H,W,C = nimg.shape
    sfactor = source_res/target_res #(origin_resolution/target_resolution)
    alg_scale = 2
    
    # use same downscaling model as the one used during training!
    nimg = generate_low_resolution_image(nimg, scale=sfactor) # scale = (source resolution / target resolution)

    print(nimg.shape)
    nimg = nimg.astype(np.float)
    nimg /= 255

    if nimg is not None:
        print(nimg.shape)
        nimg = cv2.copyMakeBorder(nimg, padding, padding, padding, padding, 
                              cv2.BORDER_REPLICATE)
        # inference -> (todo?) replace nimg by modified (iq_tool_box modifiers) instead of doing that on inference, or either use torch transforms code from modifiers (blur, sharpness, etc)
        result = inference_model(model, nimg,
                                 wind_size=wind_size, stride=stride,
                                 scale=alg_scale, batch_size=batch_size, manager=manager, add_noise=None) # you can add noise during inference to get smoother results (try from 0.1 to 0.3; the higher the smoother effect!) 

        print(result.shape)
        result = result[2*padding:-2*padding,2*padding:-2*padding]
        result = cv2.convertScaleAbs(result, alpha=np.iinfo(np.uint8).max)
        result = result.astype(np.uint8)
        #H,W = result.shape[:2] # use sfactor from original image 
        
        H_t, W_t = int(H*sfactor), int(W*sfactor)
        result = cv2.resize(result, (W_t, H_t), cv2.INTER_AREA)
        name = os.path.basename(filename).split('.')[0]
        suffix = "_sr_"+str.replace(str(target_res),".","")+"m"
        cv2.imwrite(os.path.join(path_out, f"{name}{suffix}.{fmt_out}"), result)


def main():

    parser = argparse.ArgumentParser(description="Inference MSRN")
    parser.add_argument("--filename", default="data/input_inria03m.png", type=str, help="abs path to geotif")
    parser.add_argument("--gpu_device", type=str, default='0', help="flag to indicate the GPU device to use (i.e '0' to use device 0). By default None indicates CPU")
    parser.add_argument("--path_to_model_weights", type=str, default="model.pth", help="path to the model weights")
    parser.add_argument("--wind_size", type=int, default=128, help="window size GPU processing")
    parser.add_argument("--target_res", type=float, default=0.7, help="resolution output m/px")
    parser.add_argument("--source_res", type=float, default=0.3, help="resolution imput m/px")
    opt = parser.parse_args()
    
    # Load model
    model = load_msrn_model(weights_path=opt.path_to_model_weights, cuda=opt.gpu_device) # use first GPU available
    
    # Path out
    path_out = "output/"
    
    print(path_out)

    t00=time.time()
    fout = process_file(model, path_out, compress=True, target_res=opt.target_res, source_res=opt.source_res,
                          wind_size=opt.wind_size, stride=opt.wind_size-10, 
                          batch_size=1, padding=5,filename=opt.filename)

    t01=time.time()
    print( t01-t00, "fout: ", fout)
    print("DONE ",)
        


if __name__ == "__main__":
    main()
