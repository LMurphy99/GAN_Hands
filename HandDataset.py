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
    """Class for a dataset of hand images. This version should be utilised if your anno_training.pickle contains uv_vis coordinates and you want to return those coordinates when you get an item."""

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
        
        kp_coord_uv = copy.deepcopy(self.anno_all[idx]['uv_vis'][:,:2]) # pixel coords of 42 keypoints
        kp_visible = copy.deepcopy(self.anno_all[idx]['uv_vis'][:,2])
        #camera_intrinsic_matrix = self.anno_all[idx]['K']
        kp_coord_xyz = self.anno_all[idx]['xyz'] # Global XYZ coordinates
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
            
            if hand == 'right':
                kp_coord_uv = kp_coord_uv[21:]
                kp_visible = kp_visible[21:]
                kp_coord_xyz = kp_coord_xyz[21:]
                
            else:
                kp_coord_uv = kp_coord_uv[:21]
                kp_visible = kp_visible[:21]
                kp_coord_xyz = kp_coord_xyz[:21]
                flip = True
            
            kp_coord_uv -= (left,top)
            greater_min = np.all(kp_coord_uv >= 0, axis=1)
            lesser_max = np.all(kp_coord_uv <= size, axis=1)
            kp_visible = np.logical_and(greater_min, lesser_max).astype(np.float32)

            kp_coord_uv *= (64./size)
            
        #if not self.centered:
            #kp_coord_uv -= (56,56) # params for real photos
            
            #greater_min = np.all(kp_coord_uv >= 0, axis=1)
            #lesser_max = np.all(kp_coord_uv <= size, axis=1)
            #kp_visible = np.logical_and(greater_min, lesser_max).astype(np.float32)

            #kp_coord_uv *= (64,112)
        
            
        s = np.linalg.norm(kp_coord_xyz[7,:]-kp_coord_xyz[8,:]) # scale such that index finger bone length = 1
        norm_xyz = kp_coord_xyz / s
        rel_xyz = norm_xyz - norm_xyz[0] # subtract palm location such that the hand is translation invariant
        
        R_canonical, theta, psi, omega = findRotationXZY(rel_xyz[5], rel_xyz[17]) # canonical frame based on index and pinky
        xyz_canonical = np.matmul(R_canonical, rel_xyz.T).T
        
        if xyz_canonical[5,1] < 0: # ensure every hand is backside of palm, face up
            xyz_canonical = np.matmul(Rx(math.pi), xyz_canonical.T).T
            omega += math.pi
        
        if flip:
            xyz_canonical[:,2] *= -1
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        if self.normalise:
            image = NormaliseImages(image)
        
        return [image, np.vstack((xyz_canonical, [theta,psi,omega])), mask, kp_coord_uv, kp_visible] # concaternate xyz and angles into single 22*3 vector
    
    
    
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
