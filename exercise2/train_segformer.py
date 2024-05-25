
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import shutil

from dlvc.models.segformer import SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

CONFIG = {
    # ----------------- #
    # typically changed values
    # ----------------- #
    "lr": 0.00006,
    "freeze": True,  # freeze the encoder weights
    "optimizer": "adamw",
    "scheduler": "exponential",
    "load_path": "default",  # default loads the pretrained model with the same configuration
    "fine_tune": True,
    # ----------------- #
    # values that are rarely changed
    # ----------------- #
    "num_epochs": 30,
    "batch_size": 128,
    "weight_decay": 0.01,
    "gamma": 0.98,  # only used for exponential scheduler
    "momentum": 0.9,  # only used for sgd
    "dropout": None,
    "lr_last": None,  # used for some schedulers
    "val_frequency": 2,
    "warmup_steps": 5,  # only used for custom scheduler
    "grad_clipping": 1  # dont change, just prevents exploding gradients
}


def train(args):

    train_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(),
                                   v2.ToDtype(torch.long, scale=False),
                                   v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])

    train_transform_alt = v2.Compose([v2.ToImage(),
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Resize(size=(192, 192), interpolation=v2.InterpolationMode.NEAREST),
                                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transform2_alt = v2.Compose([v2.ToImage(),
                                     v2.ToDtype(torch.long, scale=False),
                                     v2.Resize(size=(192, 192), interpolation=v2.InterpolationMode.NEAREST)])

    val_transform = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.long, scale=False),
                                 v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])

    val_transform_alt = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(size=(192, 192), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform2_c = v2.Compose([v2.ToImage(),
                                   v2.ToDtype(torch.long, scale=False),
                                   v2.Resize(size=(192, 192), interpolation=v2.InterpolationMode.NEAREST)])

    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(root="data/",
                                      split="trainval",
                                      target_types='segmentation',
                                      transform=train_transform,
                                      target_transform=train_transform2,
                                      download=True)

        val_data = OxfordPetsCustom(root="data/",
                                    split="test",
                                    target_types='segmentation',
                                    transform=val_transform,
                                    target_transform=val_transform2,
                                    download=True)
    if args.dataset == "city":
        train_data = CityscapesCustom(root="data/cityscapes/",
                                      split="train",
                                      mode="fine",
                                      target_type='semantic',
                                      transform=train_transform,
                                      target_transform=train_transform2)
        val_data = CityscapesCustom(root="data/cityscapes/",
                                    split="val",
                                    mode="fine",
                                    target_type='semantic',
                                    transform=val_transform,
                                    target_transform=val_transform2)
        # img, label = train_data.__getitem__(0)
        # img = img.numpy()
        # plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.show()
        # return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepSegmenter(
        SegFormer(num_classes=3 if args.dataset == "oxford" else 19))
    
    ## for fine tuning
    if args.dataset == 'oxford' and CONFIG["fine_tune"]:
        model.load(CONFIG["load_path"], encoder_only=True)
        if CONFIG["freeze"]:
            model.net.encoder.requires_grad_(False)
    model.to(device)

    # not working atm
    if CONFIG["dropout"] is not None:
        model.net.fc.register_forward_hook(lambda m, inp, out: F.dropout(
            out, p=CONFIG["dropout"], training=m.training))

    if CONFIG["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(
        ), lr=CONFIG["lr"], amsgrad=True, weight_decay=CONFIG["weight_decay"])
    elif CONFIG["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=CONFIG["lr"], momentum=CONFIG["momentum"], weight_decay=5e-4)

    optimizer.param_groups[0]['initial_lr'] = CONFIG["lr"]
    if CONFIG["lr_last"] is not None:
        optimizer.param_groups[0]['last_lr'] = CONFIG["lr_last"]

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=255 if args.dataset == "city" else -100)

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = CONFIG["val_frequency"]

    model_save_dir = Path(args.save_dir)
    model_save_dir.mkdir(exist_ok=True)

    if CONFIG["scheduler"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["num_epochs"])
    elif CONFIG["scheduler"] == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, epochs=CONFIG["num_epochs"], steps_per_epoch=train_data.__len__())
    elif CONFIG["scheduler"] == "customscheduler":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=custom_lr_scheduler)
    elif CONFIG["scheduler"] == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=CONFIG["gamma"])

    trainer = ImgSemSegTrainer(model,
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
    args.dataset = "oxford"  

    config_str = args.dataset + "_"
    config_str += CONFIG["optimizer"] + "_lr_" + str(CONFIG["lr"]) + "_"
    if CONFIG["optimizer"] == "sgd" and CONFIG["momentum"] is not None:
        config_str += "mom_" + str(CONFIG["momentum"]) + "_"
    config_str += CONFIG["scheduler"] + "_"
    if CONFIG["scheduler"] == "exponential" and CONFIG["gamma"] is not None:
        config_str += "gamma_" + str(CONFIG["gamma"]) + "_"
    config_str += "ep_" + str(CONFIG["num_epochs"])
    if CONFIG["dropout"] is not None:
        config_str += "_drop_" + str(CONFIG["dropout"])
    if CONFIG["weight_decay"] is not None:
        config_str += "_wd_" + str(CONFIG["weight_decay"])

    # default load path is the pretrained model with the same configuration
    if CONFIG["load_path"] == "default" and args.dataset == "oxford":
        CONFIG["load_path"] = os.path.join(
            "training", "SegFormer", config_str.replace("oxford", "city"), "SegFormer_model.pth")
    
    if CONFIG["fine_tune"] and args.dataset == "oxford":
        config_str += "_fine_tune"
    if CONFIG["freeze"] and args.dataset == "oxford":
        config_str += "_freeze"

    args.save_dir = os.path.join("training", "SegFormer", config_str)
    os.makedirs(args.save_dir, exist_ok=True)
    destination = os.path.join(args.save_dir, "train.py")
    shutil.copy(__file__, destination)

    train(args)
