import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os
import shutil

from dlvc.models.class_model import DeepClassifier 
from torchvision.models import resnet18
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset

CONFIG_NAME = "initial_config"

def train(args):
    train_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_data = CIFAR10Dataset(
        "cifar-10-batches-py", Subset.TRAINING, transform=train_transform)

    val_data = CIFAR10Dataset("cifar-10-batches-py",
                              Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepClassifier(resnet18(num_classes=10))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path(args.save_dir)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    trainer = ImgClassificationTrainer(model,
                                       optimizer,
                                       loss_fn,
                                       lr_scheduler,
                                       train_metric,
                                       val_metric,
                                       train_data,
                                       val_data,
                                       device,
                                       args.num_epochs,
                                       model_save_dir,
                                       batch_size=128,  
                                       val_frequency=val_frequency)
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    args.save_dir = os.path.join("tested_configs", "resnet18", CONFIG_NAME)
    os.makedirs(args.save_dir, exist_ok=True)
    destination = os.path.join(args.save_dir, "train.py")
    shutil.copy(__file__, destination)

    train(args)
