import os

import numpy as np
import segmentation_models_pytorch as smp
import pydensecrf.densecrf as dcrf
import torch

import unet

def load_model(name, pretrained=False):
    '''
    Loads model into memory

    Arguements:
    name (string) -- codename of the model. Avalible models:
            * pure_unet - Simple U-Net model
            * unet_resnet18 - U-Net with pretrained ResNet18 encoder
            * unet_vgg11 - U-Net with pretrained VGG11 encoder
            * unet_seresnext50 - U-Net with pretrained SE-ResNeXt50 encoder
    pretrained (boolean) -- load pretrained weights. 
                            If True, model loads in evaluation mode.
    '''     
    if not pretrained:
        if name=="pure_unet":
            model = unet.unet_model.UNet(3, 1)
        elif name=="unet_resnet18":
            model = smp.Unet("resnet18", "imagenet")
        elif name=="unet_vgg11":
            model = smp.Unet("vgg11", "imagenet")
        elif name=="se_resnext50":
            model = smp.Unet("se_resnext50_32x4d", "imagenet")
    else:
        if name=="pure_unet":
            model = torch.load('../models/best_model_unet.pth')
        elif name=="unet_resnet18":
            model = torch.load('../models/best_model_unet_reanet18_aug.pth')
        elif name=="unet_vgg11":
            model = torch.load('../models/best_model_unet_vgg11_aug.pth')
        elif name=="se_resnext50":
            model = torch.load('../models/best_model_seresnext50_aug.pth')
        model = model.eval()
    return model

class CRFModel:
    '''
    Wrapper of DenseCRF postprocessor on PyTorch segmentation models

    Arguements:
    base_model (torch.Module) -- wrapped PyTorch model
    device (string) -- device where model is held

    Methods:
    __call__(tensor) -- interface for base_model.forward(tensor)
    get_mask(image) -- apply model with DenseCRF postprocessing
                        to an input image
    '''
    def __init__(self, base_model, device=None):
        self.base = base_model
        self.device = device
        if device is not None:
            self.base = self.base.to(device)

    def __call__(self, input):
        return self.base(input)

    def get_mask(self, image, no_crf=False):
        # transform image to uint8 array for pydencecrf
        if image.dtype in {np.float16, np.float32, np.float64}:
            image = (image*255).astype(np.np.uint8)
        # get tensor for model
        tensor = self.__img_to_torch(image)
        mask = self.base(tensor)
        mask = self.__mask_to_numpy(mask)
        if no_crf:
            return mask
        # Apply DenseCRF
        mask = self.__dense_crf(image, mask)
        return mask

    def __img_to_torch(self, image):
        '''
        Tranforms image to PyTorch format

        Eg. image(240,320,3) -> tensor(1,3,240,320)
        '''
        image = image.astype(np.float32)#/255
        image = np.moveaxis(image, 2, 0)
        tensor = torch.tensor(image[np.newaxis, :, :, :])
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def __mask_to_numpy(self, tensor):
        '''
        Transforms model output to numpy array
        '''
        tensor = tensor.cpu().detach()
        if torch.min(tensor)<0 or torch.max(tensor)>1:
            tensor = torch.sigmoid(tensor)
        return tensor.numpy()[0,0]

    def __dense_crf(self, img, output_probs):
        # code from: https://github.com/milesial/Pytorch-UNet/blob/master/utils/crf.py
        h = output_probs.shape[0]
        w = output_probs.shape[1]

        output_probs = np.expand_dims(output_probs, 0)
        output_probs = np.append(1 - output_probs, output_probs, axis=0)

        d = dcrf.DenseCRF2D(w, h, 2)
        U = -np.log(output_probs)
        U = U.reshape((2, -1))
        U = np.ascontiguousarray(U)
        img = np.ascontiguousarray(img)

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=20, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

        Q = d.inference(5)
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

        return Q
