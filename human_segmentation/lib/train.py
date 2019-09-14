import os

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import segmentation_models_pytorch as smp

import data
import models

class Trainer:

    def __init__(self, model, checkpoint_path, 
                    data_train, masks_train,
                    data_valid, masks_valid,
                    preprocessing=None,
                    augmentations=None,
                    optimizer=None,
                    data_test=None, masks_test=None,
                ):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.optimizer = optimizer

        self.train = data.Dataset(data_train, masks_train,
                                    preprocessing=preprocessing,
                                    augmentation=augmentations)
        self.valid = data.Dataset(data_valid, masks_valid)
        if data_test:
            self.test = data.Dataset(data_test, masks_test)
        else: self.test=None
        
    def fit(self, epo, batch_size, verbose=0):

        # defime metrics
        criterion = smp.utils.losses.BCEDiceLoss()
        metrics = [
            smp.utils.metrics.IoUMetric(),
            smp.utils.metrics.FscoreMetric()
        ]
        
        # configure data loaders
        train_loader = DataLoader(self.train, batch_size=16, 
                                    shuffle=True, num_workers=0)
        val_loader = DataLoader(self.valid, batch_size=1, 
                                shuffle=False, num_workers=0)
        
        # configure default optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # configure training process
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=criterion, 
            metrics=metrics, 
            optimizer=optimizer,
            device=DEVICE,
            verbose=True if verbose==2 else False,
        )
        val_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=criterion, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True if verbose==2 else False,
        )

        # training loop
        iterator = range(epo) if verbose!=1 else tqdm(range(epo))
        for i in iterator:
            if verbose==2:
                print(f'\nEpoch: {i}')

            train_logs = train_epoch.run(train_loader)
            valid_logs = val_epoch.run(val_loader)

            # save best model
            if max_score < valid_logs['iou']:
                max_score = valid_logs['iou']
                torch.save(model, self.checkpoint_path)
                if verbose==2
                    print('Model saved!')
        print(f"Training completed sucessfully.\nBest model IoU: {max_score}")

    def get_validation_score(self):

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
            model, 
            loss=criterion, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True
        )
        val_epoch.run(val_loader)
        