import numpy as np 
import os
from skimage.transform import rescale

def rescaleImage(im, scale):
    #skimage.traform function
   # im = rescale(im, 1.0 / 2.0, order = 0, preserve_range = True, anti_aliasing = False)
    im = rescale(im, 1.0 / scale, order = 0, preserve_range = True, anti_aliasing = False)
    imarray = np.array(im)
    imarray = imarray.astype('float32')
    return imarray

# normalize 
def normalize_MeanSD(im):
    mean = np.mean(im)
    sd = np.std(im)

    im -= mean
    im /= sd
    return im

def normalize(im, bflag = False):
    max = np.max(im)
    min = np.min(im)

    im -= min
    im /= (max - min)

    if bflag:
        im = 2.0*im - 1.0

    return im
