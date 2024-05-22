import collections
import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass


class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int,
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5,
                 grad_clipping=None) -> None:
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.grad_clipping = grad_clipping

        self.subtract_one = isinstance(train_data, OxfordPetsCustom)

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False)

        self.one_cycle = type(
            self.lr_scheduler) == torch.optim.lr_scheduler.OneCycleLR

        self.writer = SummaryWriter(log_dir=training_save_dir)

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        loss_list = []
        self.model.train()
        self.train_metric.reset()
        for _, batch in tqdm(enumerate(self.train_loader), desc="train", total=len(self.train_loader)):
            self.optimizer.zero_grad()

            images, labels = batch
            labels = labels.squeeze(1) - int(self.subtract_one)
            images, labels = images.to(self.device), labels.to(self.device)

            predictions = self.model(images)
            if isinstance(predictions, collections.OrderedDict):
                predictions = predictions['out']

            loss = self.loss_fn(predictions, labels)
            loss.backward()

            if self.grad_clipping is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.grad_clipping)
            self.optimizer.step()

            self.train_metric.update(predictions, labels)
            loss_list.append(loss.item())

            if self.one_cycle:
                self.lr_scheduler.step()

        if not self.one_cycle:
            self.lr_scheduler.step()

        print(f"Epoch {epoch_idx}")
        print(f"Loss: {round(np.mean(loss_list), 4)}")
        print(self.train_metric)
        return np.mean(loss_list), self.train_metric.mIoU()

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """

        self.model.eval()
        with torch.no_grad():
            loss_list = []
            self.val_metric.reset()
            for _, batch in tqdm(enumerate(self.val_loader), desc="eval", total=len(self.val_loader)):
                inputs, labels = batch
                labels = labels.squeeze(1) - int(self.subtract_one)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predictions = self.model(inputs)
                if isinstance(predictions, collections.OrderedDict):
                    predictions = predictions['out']

                loss = self.loss_fn(predictions, labels)

                self.val_metric.update(predictions, labels)
                loss_list.append(loss.item())

            print(f"Validation Epoch {epoch_idx}")
            print(f"Validation Loss: {round(np.mean(loss_list), 4)}")
            print(self.val_metric)
            return np.mean(loss_list), self.val_metric.mIoU()

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        than currently saved best mean IoU or if it is end of training. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        best_val_mIoU = 0
        for epoch in range(1, self.num_epochs + 1):
            loss_train, mIoU_train = self._train_epoch(epoch)

            if epoch % self.val_frequency == 0:
                loss_val, mIoU_val = self._val_epoch(epoch)

                self.write_to_tensorboard(epoch, loss_val, mIoU_val, "val")

                if mIoU_val > best_val_mIoU:
                    best_val_mIoU = mIoU_val
                    self.model.save(self.training_save_dir)

            self.write_to_tensorboard(
                epoch, loss_train, mIoU_train, "train", self.optimizer.param_groups[0]['lr'])

        self.writer.close()

    def write_to_tensorboard(self, epoch, loss, mIoU, mode, lr=None):
        self.writer.add_scalar(f'Loss/{mode}', loss, epoch)
        self.writer.add_scalar(f'mIoU/{mode}', mIoU, epoch)
        if lr:
            self.writer.add_scalar(f'LearningRate/{mode}', lr, epoch)
