import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# for wandb users:
# from dlvc.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds training logic for one epoch.
        '''

        pass


class ImgClassificationTrainer(BaseTrainer):
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
                 val_frequency: int = 5) -> None:
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
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

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False)
        
        self.writer = SummaryWriter()

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        print(f"Epoch {epoch_idx}")
        loss_list = []
        self.model.train()
        self.train_metric.reset()
        for batch, labels in self.train_loader:
            labels = labels.type(torch.LongTensor)
            batch, labels = batch.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch)
            loss = self.loss_fn(predictions, labels)
            loss.backward()
            self.optimizer.step()

            self.train_metric.update(predictions, labels)
            loss_list.append(loss.item())
        self.lr_scheduler.step()
        print(f"Loss: {np.mean(loss_list)}")
        print(self.train_metric.__str__())
        return np.mean(loss_list), self.train_metric.accuracy(), self.train_metric.per_class_accuracy()

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        print(f"Validation Epoch {epoch_idx}")
        self.model.eval()
        with torch.no_grad():
            loss_list = []
            self.val_metric.reset()
            for batch, labels in self.val_loader:
                labels = labels.type(torch.LongTensor)
                batch, labels = batch.to(self.device), labels.to(self.device)

                predictions = self.model(batch)
                loss = self.loss_fn(predictions, labels)

                self.val_metric.update(predictions, labels)
                loss_list.append(loss.item())
            print(f"Validation Loss: {np.mean(loss_list)}")
            print(self.val_metric.__str__())
            return np.mean(loss_list), self.val_metric.accuracy(), self.val_metric.per_class_accuracy()

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        best_val_pc_acc = 0
        for epoch in range(self.num_epochs):
            loss_train, acc_train, pc_acc_train = self._train_epoch(epoch)
            if (epoch+1) % self.val_frequency == 0:
                loss_val, acc_val, pc_acc_val = self._val_epoch(epoch)
                if pc_acc_val > best_val_pc_acc:
                    best_val_pc_acc = pc_acc_val
                    self.model.save(self.training_save_dir,
                                    suffix=f"epoch_{epoch}")
                self.writer.add_scalar('Loss/val', loss_val, epoch)
                self.writer.add_scalar('Accuracy/val', acc_val, epoch)
                self.writer.add_scalar('PerClassAcc/val', pc_acc_val, epoch)

            self.writer.add_scalar('Loss/train', loss_train, epoch)
            self.writer.add_scalar('Accuracy/train', acc_train, epoch)
            self.writer.add_scalar('PerClassAcc/train', pc_acc_train, epoch)
            print("\n")
