import os

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

from lib import data
from lib import models

class Trainer:
    '''
    Trainer for PyTorch models

    Arguements:
    model (toch.Module) -- Torch model
    checkpoint_path (str) -- Path to save checkpoints
    data_train(valid/test) -- Path to data directory
    masks_train(vaid/test) -- Path to masks
    preprocessing -- Preprocessing function
    augmentations -- Augmentation function
    optimizer -- Custom optimizer for model

    Methods:
    fit(epo, batch_size, verbose) -- training function
    get_test_score() -- get score on test data if avalible 
    '''
    def __init__(self, model, checkpoint_path, 
                    data_train, masks_train,
                    data_valid, masks_valid,
                    preprocessing=None,
                    augmentations=None,
                    optimizer=None, device="cpu",
                    data_test=None, masks_test=None,
                ):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.optimizer = optimizer
        self.device = device

        self.train = data.Dataset(data_train, masks_train,
                                    preprocessing=preprocessing,
                                    augmentation=augmentations)
        self.valid = data.Dataset(data_valid, masks_valid)
        if data_test:
            self.test = data.Dataset(data_test, masks_test)
        else: self.test=None
        
    def fit(self, epo, batch_size, verbose=0):
        '''
        Fit model

        Arguements:
        epo (integer) -- Numbers of epochs
        batch_size (integer) -- Number of images in training data batch
        verbose (integer) -- 0 for no output
                             1 for only epo output
                             2 for full output
        '''
        # defime metrics
        criterion = smp.utils.losses.BCEDiceLoss()
        metrics = [
            smp.utils.metrics.IoUMetric(),
            smp.utils.metrics.FscoreMetric()
        ]
        
        # configure data loaders
        train_loader = DataLoader(self.train, batch_size=batch_size, 
                                    shuffle=True, num_workers=0)
        val_loader = DataLoader(self.valid, batch_size=1, 
                                shuffle=False, num_workers=0)
        
        # configure default optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # configure training process
        train_epoch = smp.utils.train.TrainEpoch(
            self.model, 
            loss=criterion, 
            metrics=metrics, 
            optimizer=self.optimizer,
            device=self.device,
            verbose=True if verbose==2 else False,
        )
        val_epoch = smp.utils.train.ValidEpoch(
            self.model, 
            loss=criterion, 
            metrics=metrics, 
            device=self.device,
            verbose=True if verbose==2 else False,
        )

        # training loop
        logs = []
        max_score = 0
        iterator = range(epo) if verbose!=1 else tqdm(range(epo))
        for i in iterator:
            if verbose==2:
                print(f'\nEpoch: {i}')

            train_logs = train_epoch.run(train_loader)
            valid_logs = val_epoch.run(val_loader)
            
            logs.append((train_logs, valid_logs))

            # save best model
            if max_score < valid_logs['iou']:
                max_score = valid_logs['iou']
                torch.save(self.model, self.checkpoint_path)
                if verbose==2:
                    print('Model saved!')
        print(f"Training completed sucessfully.\nBest model IoU: {max_score}")
        return max_score, logs

    def get_test_score(self):
        '''
        Calculate test score
        '''
        if self.test is None:
            raise ValueError("No test data provided")
        # defime metrics
        criterion = smp.utils.losses.BCEDiceLoss()
        metrics = [
            smp.utils.metrics.IoUMetric(),
            smp.utils.metrics.FscoreMetric()
        ]
        # configure data loader
        val_loader = DataLoader(self.test, batch_size=1, 
                                shuffle=False, num_workers=0)
        # perform testing
        val_epoch = smp.utils.train.ValidEpoch(
            self.model, 
            loss=criterion, 
            metrics=metrics, 
            device=self.device,
            verbose=True
        )
        score = val_epoch.run(val_loader)
        return score
