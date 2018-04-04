import numpy as np
from numpy import matlib


def make_orientation_image(theta,t = 0,omega = 0.08,size = 227):
    # make orientation image
    x = np.arange(-size/2.0,size/2.0)
    y = np.arange(-size/2.0,size/2.0)

    xx = matlib.repmat(x,len(x),1)
    yy = matlib.repmat(y,len(y),1).T
    
    xy_rotation = xx*np.cos(theta)-yy*np.sin(theta)
    wave = np.cos(xy_rotation*omega+t)
    temp = wave - np.min(wave)
    wave = temp / np.max(temp)*255
    wave = np.stack([wave]*3,axis=2).astype(np.uint8)
    return(wave)
