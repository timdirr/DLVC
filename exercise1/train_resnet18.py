import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os
import shutil

from dlvc.models.class_model import DeepClassifier  
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from torchvision.models import resnet18
import torch.nn.functional as F

CONFIG = {
    "lr": 0.002,
    "lr_last": 0.0001, 
    "num_epochs": 100,
    "batch_size": 256,
    "grad_clipping": 1,
    "val_frequency": 5,
    "dropout": 0.2,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "weight_decay": 0.1,
    "modified": True, 
    "momentum": None, 
    "warmup_steps": 5 # only used for custom scheduler
}

def train(args):
    train_transform = v2.Compose([v2.ToImage(),
                                  v2.RandomHorizontalFlip(p=0.5),
                                  v2.RandomCrop(32, padding=4, padding_mode='reflect'), 
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_data = CIFAR10Dataset("cifar-10-batches-py", Subset.TRAINING, transform=train_transform)

    val_data = CIFAR10Dataset("cifar-10-batches-py", Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepClassifier(resnet18(num_classes=10))
    if CONFIG["modified"]:
        model.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.to(device)

    if CONFIG["dropout"] is not None:
        model.net.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=CONFIG["dropout"], training=m.training))

    if CONFIG["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], amsgrad=True, weight_decay=CONFIG["weight_decay"])
    elif CONFIG["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["lr"], momentum=0.9, weight_decay=5e-4)

    optimizer.param_groups[0]['initial_lr'] = CONFIG["lr"]
    optimizer.param_groups[0]['last_lr'] = CONFIG["lr_last"]

    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path(args.save_dir)

    if CONFIG["scheduler"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    elif CONFIG["scheduler"] == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=CONFIG["num_epochs"], steps_per_epoch=train_data.__len__())
    elif CONFIG["scheduler"] == "customscheduler":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_lr_scheduler)

    trainer = ImgClassificationTrainer(model,
                                       optimizer,
                                       loss_fn,
                                       lr_scheduler,
                                       train_metric,
                                       val_metric,
                                       train_data,
                                       val_data,
                                       device,
                                       CONFIG["num_epochs"],
                                       model_save_dir,
                                       batch_size=CONFIG["batch_size"], 
                                       grad_clipping=CONFIG["grad_clipping"],
                                       val_frequency=val_frequency) 
    trainer.train()

def custom_lr_scheduler(current_step: int):
    warmup_steps = CONFIG["warmup_steps"]
    if current_step < warmup_steps:  
        return float(current_step / warmup_steps)
    else:
        return max(CONFIG["lr_last"], float(CONFIG["num_epochs"] - current_step) / float(max(1, CONFIG["num_epochs"] - warmup_steps)))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    config_str = ""
    if CONFIG["modified"]:
        config_str = "mod_"
    config_str += CONFIG["optimizer"] + "_lr_" + str(CONFIG["lr"]) + "_"
    if CONFIG["optimizer"] == "sgd" and CONFIG["momentum"] is not None:
        config_str += "mom_" + str(CONFIG["momentum"]) + "_"
    config_str += CONFIG["scheduler"] + "_"
    config_str += "ep_" + str(CONFIG["num_epochs"])
    if CONFIG["grad_clipping"] is not None:
        config_str += "_gclip_" + str(CONFIG["grad_clipping"])
    if CONFIG["dropout"] is not None:
        config_str += "_drop_" + str(CONFIG["dropout"])
    if CONFIG["weight_decay"] is not None:
        config_str += "_wd_" + str(CONFIG["weight_decay"])

    args.save_dir = os.path.join("advanced_testing", "resnet18", config_str)
    os.makedirs(args.save_dir, exist_ok=True)
    destination = os.path.join(args.save_dir, "train.py")
    shutil.copy(__file__, destination)

    train(args)
