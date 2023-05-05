import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from pathlib import Path
from PIL import Image, ImageOps
import os
from math import ceil, floor
from pandas import DataFrame
from sklearn.cluster import KMeans
import sys
import argparse

from collections import OrderedDict

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def readonly_handler(func, path, execinfo): 
    os.chmod(path, 128)
    func(path)

# defining the function to ignore the files
# if present in any folder
def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    
def load_temp_bmp(path):
    I = Image.open(path).convert('L')
    I = np.fromfile(path, dtype=np.uint16, offset=66)
    I = I.reshape((220,174))
    
    ol = (24,11,)
    dim = (175,150,)
    I = np.array(I)[ol[0]:ol[0]+dim[0], ol[1]:ol[1]+dim[1]]
    I = I.astype(np.uint16)
    return I

def getList(dict):
    return dict.keys()

#find cross
def findCross(I):
    crossRow = []
    crossCol = []
    [m,n] = I.shape
    kors = np.array([0,2**16-1,2**16-1,2**16-1,2**16-1,2**16-1,2**16-1,2**16-1,0])
    korss = np.concatenate((kors, kors))
    length = 2*len(kors)+5
    idx = tuple(range(0,9)) + tuple(range(14,23))
    for k in tuple(range(ceil(m/3), m)) + tuple(range(1, floor(m/3))):
        for kk in range(0, n - length):
            idx2 = [z+kk for z in idx]
            tmp = I[k, idx2]
            if np.all(tmp == korss):
                crossRow = k
                crossCol = kk+11
    return(crossRow, crossCol)
    
def median_NSEW( I ):
    medianColor = np.median(np.concatenate([I[0,2][None,], I[1,:], I[11,:], I[2:7, 0], I[2:7, 4]]))
    I[2:8, 1:4] = medianColor
    I[8:11, :] = medianColor
    return I

def median_DIAG( I ):
    medianColor = np.median(np.concatenate([I[4,1][None,], I[0,1:5], I[5, 1:5], I[:,0], I[:, 5]]))
    I[1:4, 1:5] = medianColor
    I[4, 2:5] = medianColor
    return I
    
def fill_cross(I):
    #fill cross with color
    cross = findCross(I)

    #North
    N = I[cross[0]-13:cross[0]-1, cross[1]-2:cross[1]+3]
    N = median_NSEW(N)
    I[cross[0]-13:cross[0]-1, cross[1]-2:cross[1]+3] = N

    #South
    S = I[cross[0]+2:cross[0]+14, cross[1]-2:cross[1]+3]
    S = median_NSEW(np.flip(S))
    I[cross[0]+2:cross[0]+14, cross[1]-2:cross[1]+3] = np.flip(S)

    #East
    E = I[cross[0]-2:cross[0]+3, cross[1]+1:cross[1]+13]
    E = median_NSEW(np.rot90(E, 3))
    I[91:96, 72:84] = np.rot90(E,1)

    #West
    W = I[cross[0]-2:cross[0]+3, cross[1]-12:cross[1]]
    W = median_NSEW(np.rot90(W, 1))
    I[cross[0]-2:cross[0]+3, cross[1]-12:cross[1]] = np.rot90(W, 3)

    #North- East
    NE = I[cross[0]-6:cross[0], cross[1]+1:cross[1]+7]
    NE = median_DIAG(NE)
    I[cross[0]-6:cross[0], cross[1]+1:cross[1]+7] = NE

    #North- West
    NW = I[cross[0]-6:cross[0], cross[1]-6:cross[1]]
    NW = median_DIAG(np.flip(NW))
    I[cross[0]-6:cross[0], cross[1]-6:cross[1]] = np.flip(NW)

    #South- East
    SE = I[cross[0]+1:cross[0]+7, cross[1]+1:cross[1]+7]
    SE = median_DIAG(np.rot90(SE, 1))
    I[cross[0]+1:cross[0]+7, cross[1]+1:cross[1]+7] = np.rot90(SE, 3)

    #South- West
    SW = I[cross[0]+1:cross[0]+7, cross[1]-6:cross[1]]
    SW = median_DIAG(np.rot90(SW, 2))
    I[cross[0]+1:cross[0]+7, cross[1]-6:cross[1]] = np.rot90(SW, 2)
    
    return cross, I
    
def get_segmentation(image, boundaries):
    num_segments = len(boundaries)-1

    segments = np.zeros((num_segments,*image.shape))

    for i in range(num_segments):
        a,b = boundaries[i:i+2]
        segments[i,...] = np.logical_and(a<=image, image<b)
    return segments

def fuse_segments(segments, values=None):
    return np.einsum('i,ijk->jk',values,segments)
    # C[j,k] = summe_{über i} values[i]*segments[i,j,k]