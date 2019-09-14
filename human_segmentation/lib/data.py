import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import imgaug.augmenters as iaa
from torch.utils import data

class Dataset(data.Dataset):
    """
    Dataset class

    Used to create dataset from images on disk.
    Also applies preprocessing and augmentations to images.
    
    Arguements:
    img_path (string) -- path to images
    mask_path (string) -- path to masks
    size (tuple) -- reqired size of output images
    preprocessing (function(image, mask) -> image, mask) -- 
                                        preprocessing function
    augmentation (function) -- augmentation module from imgaug library
    """
    
    def __init__(self, img_path, mask_path=None, 
        size=(224, 224), 
        preprocessing=None, augmentation=None):

        # file paths to images and masks
        ids = [x.split(".")[0] for x in os.listdir(img_path)]
        self.img_paths = [os.path.join(img_path, x+".jpg") for x in ids]
        self.mask_path = mask_path
        if mask_path is not None:
            self.mask_paths = [os.path.join(mask_path, x+".png") for x in ids]
        
        self.size = size
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        
    def __getitem__(self, i):
        # load images and resize to required size
        img = plt.imread(self.img_paths[i])
        img = cv2.resize(img, self.size)
        # load masks if needed
        if self.mask_path is not None:
            mask = plt.imread(self.mask_paths[i])
            mask = cv2.resize(mask, self.size)
        
        # apply preprocessing if needed
        if self.preprocessing:
            img, mask = self.preprocessing(image=img, mask=mask)
    
        # apply augmentation if needed
        if self.augmentation:
            img, mask = self.augmentation(image=img, 
                                            segmentation_maps=mask)
        
        # translate image to pytorch format
        img = np.rollaxis(img, 2, 0)
        img = img.astype(np.float32)
        
        # translate mask to PyTorch format
        if self.mask_path is not None:
            mask = mask[np.newaxis, :, :].astype(np.float32)
            
        if self.mask_path is not None:
            return img, mask
        else: return img
            
    def __len__(self):
        return len(self.img_paths)

def load_augmentations(flip=0.5, blur=0.2, crop=0.5,
                        contrast=0.3, elstic=0.2,
                        affine=0.5):
    """
    Loads and configures data augmenter object

    Arguements:
    flip (float) -- probality of horizontal flip
    crop (float) -- probability of random crop
    blur (float) -- probability of gaussian blur
    contrast (float) -- probability of pixelwise color transformation
    elastic (float) -- probability of elastic distortion
    affine (float) -- probability of affine transform
    noise (float) -- probability of one of noises
    """
    aug = iaa.Sequential([
            iaa.Fliplr(flip),

            iaa.Sometimes(crop, 
                iaa.Crop(px=(0, 20))
            ),

            iaa.Sometimes(blur, 
                iaa.GaussianBlur(sigma=(0.5, 5))
            ),

            iaa.Sometimes(contrast, 
                iaa.SomeOf(
                    (1,5),
                    [
                        iaa.GammaContrast(per_channel=True, gamma=(0.25,1.75)),
                        iaa.LinearContrast(alpha=(0.25,1.75), per_channel=True),
                        iaa.HistogramEqualization(to_colorspace="HLS"),
                        iaa.LogContrast(gain=(0.5, 1.0)),
                        iaa.CLAHE(clip_limit=(1,10))
                    ]
                ),
            ),

            iaa.Sometimes(elastic, 
                iaa.OneOf([
                    iaa.ElasticTransformation(alpha=20, sigma=1),
                    iaa.ElasticTransformation(alpha=200, sigma=20)
                ])
            ),

            iaa.Sometimes(affine, 
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    rotate=(-30, 30),
                    order=[0, 1]
                )
            )
        ])
    return aug
    