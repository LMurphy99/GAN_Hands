import os
import scipy.ndimage as ndi
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
from Rotations import *
import math
import matplotlib.pyplot as plt
import copy
try:
    from google.colab import drive
    import pickle5 as pickle
except:
    import pickle

def NormaliseImages(X):
    X = X - X.mean(dim=(1,2))[:,None][:,None]
    X /= X.std(dim=(1,2))[:,None][:,None]
    X = torch.sigmoid(X)
    numerator = X - X.amin(dim=(1,2))[:,None][:,None]
    denominator = X.amax(dim=(1,2))[:,None][:,None] - X.amin(dim=(1,2))[:,None][:,None]
    return ((numerator*2) / denominator)-1

def NormaliseBatch(X):
    X = X - X.mean(dim=(2,3))[:,None][:,None].permute(0,3,1,2)
    X /= X.std(dim=(2,3))[:,None][:,None].permute(0,3,1,2)
    X = torch.sigmoid(X)
    numerator = X - X.amin(dim=(2,3))[:,None][:,None].permute(0,3,1,2)
    denominator = X.amax(dim=(2,3))[:,None][:,None].permute(0,3,1,2) - X.amin(dim=(2,3))[:,None][:,None].permute(0,3,1,2)
    return ((numerator*2) / denominator)-1

class HandDataset(Dataset):
    """Version of the Hand Dataset which does not return coordinates, only the image and mask of a hand."""
    
    def __init__(self, img_dir, mask_dir, anno_path, black_size, zfill, ext, transform=None, masked=False, centered=False, coef=2.35, normalise=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.masked = masked
        self.black = Image.new('RGB', (black_size, black_size))
        self.centered = centered
        self.coef = coef # coefficient from which to scale image crops
        self.zfill = zfill # number of digits in image name
        self.ext = ext # file extension for images
        self.normalise = normalise
        
        with open(anno_path, 'rb') as file:
            self.anno_all = pickle.load(file)
        
    def __len__(self):
        return len(os.listdir(self.mask_dir))
        
    
    def __getitem__(self, idx):
        file = str(idx).zfill(self.zfill)+self.ext
        image = Image.open( os.path.join(self.img_dir, file) )
        mask = Image.open( os.path.join(self.mask_dir, file) ).convert('L')
        flip = False # whether to flip z axis
        
        if self.masked and not self.centered: # for typical purposes, this will trigger for real hands
            image = Image.composite(image, self.black, mask)
        
        if self.centered: # for typical purposes, this will trigger for synthetic hands
            mask_right = mask.point(lambda p: p>= 18) # mask for right hand
            mask_left = mask.point(lambda p: p>=2 and p<=17) # mask for left hand
            std_right = ndi.standard_deviation(np.array(mask_right)) # std of right mask
            std_left = ndi.standard_deviation(np.array(mask_left)) # std of left mask
            
            if std_right > std_left: # based on assumption that higher std implies clearer image of hand...
                mask = mask_right.point(lambda p: p > 0 and 255)
                hand = 'right'
            else:
                mask = mask_left.point(lambda p: p > 0 and 255)
                hand = 'left'
            
            std = ndi.standard_deviation(np.array(mask)) # variables used to crop and scale new image.
            center = ndi.center_of_mass(np.array(mask)) 

            if std == 0:
                raise Exception(f'help! No hand visible! Check idx {idx}')
                
            if self.masked:
                image = Image.composite(image, self.black, mask)
                
            d = std * self.coef / 2 #distance from center
            size = int(std * self.coef) #pixel width & height
            top = int(center[0]-d)
            left = int(center[1]-d)
            image = transforms.functional.crop(image, top, left, size, size)
            mask = transforms.functional.crop(mask, top, left, size, size)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        if self.normalise:
            image = NormaliseImages(image)
        
        return [image, -1, mask, -1, -1] # -1 in positions of coordinates
    
    
    
colours = ['r','b','k','brown','g']

def plotHand2D(x, y):
    for i, c in zip(range(5), colours):
        for j in range(i*4+1, i*4+4):
            plt.plot([x[j],x[j+1]], [y[j], y[j+1]], c=c)
    
    plt.plot([x[0], x[1]], [y[0], y[1]], c='y')
    plt.plot([x[0], x[17]], [y[0], y[17]], c='y')
    
    for i in range(4):
        plt.plot([x[i*4+1], x[i*4+5]], [y[i*4+1], y[i*4+5]], c='y')

        
def plotHand3D(ax, x, y, z):
    for i, c in zip(range(5), colours):
        for j in range(i*4+1, i*4+4):
            ax.plot([x[j],x[j+1]], [y[j], y[j+1]], [z[j], z[j+1]], c=c)
    
    ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], c='y')
    ax.plot([x[0], x[17]], [y[0], y[17]], [z[0], z[17]], c='y')
    ax.plot([x[1], x[5]], [y[1], y[5]], [z[1], z[5]], c='y')
    
    for i in range(1,4):
        ax.plot([x[i*4+1], x[i*4+5]], [y[i*4+1], y[i*4+5]], [z[i*4+1], z[i*4+5]], c='y')
