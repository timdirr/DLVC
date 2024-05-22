
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.segformer import SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
import numpy as np


def train(args):

    train_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(
                                      size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(),
                                   v2.ToDtype(torch.long, scale=False),
                                   # ,
                                   v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])

    val_transform = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Resize(
                                    size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.long, scale=False),
                                 v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)])

    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(root="path_to_dataset",
                                      split="trainval",
                                      target_types='segmentation',
                                      transform=train_transform,
                                      target_transform=train_transform2,
                                      download=True)

        val_data = OxfordPetsCustom(root="path_to_dataset",
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
        img, label = train_data.__getitem__(0)
        img2, label2 = val_data.__getitem__(1)
        label = label.numpy()
        label[label == 255] = 0
        print(label.min(), label.max())
        img = img.numpy()
        img2 = img2.numpy()
        b = [img, img2]
        b = np.array(b)
        print(b.shape)
        x = np.array(np.argmax(b, axis=1))
        print(x.shape)
        print(x)
        print(x.min(), x.max())

    # device = ...

    # model = DeepSegmenter(...)
    # # If you are in the fine-tuning phase:
    # if args.dataset == 'oxford':
    #     ##TODO update the encoder weights of the model with the loaded weights of the pretrained model
    #     # e.g. load pretrained weights with: state_dict = torch.load("path to model", map_location='cpu')
    #     ...
    #     ##
    # model.to(device)
    # optimizer = ...
    # loss_fn = ... # remember to ignore label value 255 when training with the Cityscapes datset

    # train_metric = SegMetrics(classes=train_data.classes_seg)
    # val_metric = SegMetrics(classes=val_data.classes_seg)
    # val_frequency = 2 # for

    # model_save_dir = Path("saved_models")
    # model_save_dir.mkdir(exist_ok=True)

    # lr_scheduler = ...

    # trainer = ImgSemSegTrainer(model,
    #                 optimizer,
    #                 loss_fn,
    #                 lr_scheduler,
    #                 train_metric,
    #                 val_metric,
    #                 train_data,
    #                 val_data,
    #                 device,
    #                 args.num_epochs,
    #                 model_save_dir,
    #                 batch_size=64,
    #                 val_frequency = val_frequency)
    # trainer.train()
    # # see Reference implementation of ImgSemSegTrainer
    # # just comment if not used
    # trainer.dispose()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31
    args.dataset = "city"

    train(args)
