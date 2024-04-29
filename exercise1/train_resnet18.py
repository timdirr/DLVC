# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os
import shutil

from dlvc.models.class_model import DeepClassifier  # etc. change to your model
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from torchvision.models import resnet18
import torch.nn.functional as F

CONFIG_NAME = "modified_resnet18_adam_cosineannealinglr"

def train(args):
    train_transform = v2.Compose([v2.ToImage(),
                                  v2.RandomHorizontalFlip(p=0.5),
                                  v2.RandomCrop(32, padding=4, padding_mode='reflect'), 
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
    model.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.to(device)

    # model.net.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.2, training=m.training))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer.param_groups[0]['initial_lr'] = args.lr
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path(args.save_dir)

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=args.num_epochs, steps_per_epoch=train_data.__len__())
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

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
                                       batch_size=256, 
                                       # grad_clipping=0.1,
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
    args.num_epochs = 100
    args.lr = 0.001

    args.save_dir = os.path.join("tested_configs", "resnet18", CONFIG_NAME)
    os.makedirs(args.save_dir, exist_ok=True)
    destination = os.path.join(args.save_dir, "train.py")
    shutil.copy(__file__, destination)

    train(args)
