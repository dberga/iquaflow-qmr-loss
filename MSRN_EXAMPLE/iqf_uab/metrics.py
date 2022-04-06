import os
import sys
import piq
fid_metric = piq.FID()
import math
import torch
import numpy as np
import cv2

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        # clamp negative values to zero (values should range between 0. and 1.)
        if len(img1[img1<0]>0) or len(img2[img2<0]>0) or len(img1[img1>1]>0) or len(img2[img2>1]>0):
            img1p=img1.clone();
            img1p[img1p<0.]=0.
            img1p[img1p>1.]=1.
            img2p=img2.clone();
            img2p[img2p<0.]=0.
            img2p[img2p>1.]=1.
            return piq.ssim(img1p,img2p)
        return piq.ssim(img1,img2)
        '''
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[0] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1[i,:,:], img2[i,:,:]))
                return np.array(ssims).mean()
            elif img1.shape[0] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        elif img1.ndim == 4:
            N=img1.shape[0];
            for n in range(N):
                if img1.shape[1] == 3:
                    ssims = []
                    for i in range(3):
                        ssims.append(self._ssim(img1[n,i,:,:], img2[n,i,:,:]))
                    return np.array(ssims).mean()
                elif img1.shape[1] == 1:
                    return self._ssim(np.squeeze(img1[n,:,:,:]), np.squeeze(img2[n,:,:,:]))
        '''

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()

class FID:
    """Frechet inception distance
    img1, img2: [0., 1.0]"""

    def __init__(self):
        self.name = "FID"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        '''
        fimg1=fid_metric.compute_feats(img1)
        fimg2=fid_metric.compute_feats(img2)
        fid: torch.Tensor = self.fid_metric(fimg1,fimg2)
        return fid
        '''
        N=img1.shape[0];
        fids=[]
        for n in range(N):
            fid_value = np.sum( [
                fid_metric( torch.squeeze(img2)[n,i,...], torch.squeeze(img1)[n,i,...] ).item()
                for i in range( img2.shape[1] )
                ] ) / img2.shape[1]
            fids.append(fid_value)
        return np.array(fids).mean()


