import argparse

import cv2
import glob
import os
import sys
import time

import numpy as np

from utils.msrn import load_msrn_model, inference_model

from utils.utils import (read_georaster_input,
                         save_tif,
                         get_mask,
                         convert_float_uintX,
                         convert_uintX_float)



def process_file(model, path_out, compress=True, res_output=0.7,
                wind_size=512, stride=480, batch_size=1, padding=5, manager=None, filename=None):
    
    nimg = read_georaster_input(filename)
    dtype_input='uint8'
    channel_order_out = 'rgb'
    fout=None

    if nimg is not None:
        mask = get_mask(nimg)
        nimg[mask==0]=0

        nimg = cv2.copyMakeBorder(nimg, padding, padding, padding, padding, 
                              cv2.BORDER_REPLICATE)

        # inference
        result = inference_model(model, nimg,
                                 wind_size=wind_size, stride=stride,
                                 scale=2, batch_size=batch_size, manager=manager) 

        result = result[2*padding:-2*padding,2*padding:-2*padding]
        result = cv2.convertScaleAbs(result, alpha=np.iinfo(np.uint8).max)

        H,W = result.shape[:2]

        if mask is not None:
            # resize mask to fit SR
            mask = cv2.resize((mask*255).astype(np.uint8), (W, H), cv2.INTER_NEAREST)
            result = result.transpose(2,0,1)
            for i in range(result.shape[0]):
                result[i] = np.where((result[i] == 0) & (mask > 0), 1, result[i])
            result = result.transpose(1,2,0)

        name = os.path.basename(filename).split('.')[0]

        fout = save_tif(path_out, filename, result, target_resolution=res_output,
                        name = name,  name_id='sr', 
                        channel_order_out='rgb', compress=compress)
    return fout


def main():

    parser = argparse.ArgumentParser(description="Inference MSRN")
    parser.add_argument("--filename", default="data/sample_1mpx.tif", type=str, help="abs path to geotif")
    parser.add_argument("--gpu_device", type=str, default=None, help="flag to indicate the GPU device to use (i.e '0' to use device 0). By default None indicates CPU")
    parser.add_argument("--path_to_model_weights", type=str, default="model.pth", help="path to the model weights")
    parser.add_argument("--wind_size", type=int, default=512, help="window size GPU processing")
    parser.add_argument("--res_output", type=float, default=0.7, help="resolution output m/px")
    opt = parser.parse_args()
    
    # Load model
    model = load_msrn_model(weights_path=opt.path_to_model_weights, cuda=opt.gpu_device) # use first GPU available
    
    # Path out
    path_out = "output/"
    
    print(path_out)

    t00=time.time()
    fout = process_file(model, path_out, compress=True, res_output=opt.res_output,
                          wind_size=opt.wind_size, stride=opt.wind_size-10, 
                          batch_size=1, padding=5,filename=opt.filename)

    t01=time.time()
    print( t01-t00, "fout: ", fout)
    print("DONE ",)
        


if __name__ == "__main__":
    main()
