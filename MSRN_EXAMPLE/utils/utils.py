import os
import cv2
import sys
import rasterio
import numpy as np

    
def save_tif(path_out_samples, fname, res, target_resolution=0.7,
             name=None, name_id='MSRN07',
             channel_order_out='rgb', 
             compress=False, nodata=0):
    """
    input is float normalized between 0 an 1.
    by default it taked input and rescale values to targert dtype (uint8).
    
    """
    fmt_in=os.path.basename(fname).split('.')[1]
    fmt_out="tif"

    if channel_order_out=='rgb':
        res_tmp = res.copy()
        res = res[:,:,::-1]
    
    if name is None:
        name=os.path.basename(fname).split('.')[0]

    if len(name_id)>0:
        file_out = os.path.join(path_out_samples, f"TMP_{name}_{name_id}."+fmt_out)
    else:
        file_out = os.path.join(path_out_samples, f"TMP_{name}."+fmt_out)
    
    if fmt_in == "tif": # with geo data
        cmd = f"gdalwarp -tr {target_resolution} {target_resolution} \"{fname}\" \"{file_out}\"" # -r lanczos
        os.system(cmd)
        src=rasterio.open(file_out, "r")
        H_t, W_t = src.shape
        res = cv2.resize(res, (W_t, H_t), cv2.INTER_AREA)
        print(res.shape, src.read().shape)
        meta = src.meta.copy()
        meta['dtype']=res.dtype.name
        meta['count']=f'{res.shape[-1]}'
        meta['nodata']=nodata
        if compress:
            meta['compress']='lzw'
        with rasterio.open(file_out.replace('TMP_', ''), "w", **meta) as dst:
            dst.write(res.transpose(2,0,1))
        os.system(f"rm {file_out}")
    else: # without geo data
        res_factor=1/target_resolution
        H_t=round(res.shape[0]*res_factor)
        W_t=round(res.shape[1]*res_factor)
        src=cv2.resize(res, (H_t, W_t), cv2.INTER_AREA)
        cv2.imwrite(file_out.replace('TMP_',''),src[:,:,::-1])

    return file_out.replace('TMP_', '')
                

############################ RASTERS ##########################################


def read_georaster_input(fname):
    """    
    args:
        fname:   path to .tif
        
    returns:
        nimg:    numpy array H, W, C normalized to 0-1
        
    """
    
    with rasterio.open(fname, 'r') as src:
        nimg = src.read()[:3]
    nimg = nimg[::-1] # BGR for network input (cv2 style)        

    drange = np.iinfo(nimg.dtype.name).max
    # normalize to 0-1 range
    nimg = nimg.astype(np.float)/drange
    
    # H,W,C dim order
    nimg = nimg.transpose(1,2,0)
    return nimg


def get_mask(nimg):
    image = nimg.transpose(2,0,1).copy()
    mask = []
    C = image.shape[0]
    for i in range(C):
        mask.append(1*(image[i]>0))
    mask = np.array(mask)
    mask = np.sum(mask, 0)
    mask[mask!=C]=0
    mask[mask>0]=1
    return mask


def convert_float_uintX(nimg, dtype=np.uint16):
    nimg = nimg*(np.iinfo(dtype).max)
    nimg = nimg.astype(dtype)
    return nimg

def convert_uintX_float(nimg):
    dtype = nimg.dtype
    nimg = nimg.astype(np.float)
    nimg = nimg/np.iinfo(dtype).max
    return nimg
