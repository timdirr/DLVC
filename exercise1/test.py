## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os

from torchvision.models import resnet18 # change to the model you want to test
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
import numpy as np




def test(args):
    transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    test_data = CIFAR10Dataset("cifar-10-batches-py", Subset.TEST, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_data = len(test_data)

    model = DeepClassifier(resnet18(num_classes=10))
    model.load(args.path_to_trained_model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    test_metric = Accuracy(classes=test_data.classes)

    ### Below implement testing loop and print final loss 
    ### and metrics to terminal after testing is finished
    model.eval()
    with torch.no_grad():
        loss_list = []
        test_metric.reset()
        for batch, labels in test_data_loader:
            labels = labels.type(torch.LongTensor)
            batch, labels = batch.to(device), labels.to(device)

            predictions = model(batch)
            loss = loss_fn(predictions, labels)

            test_metric.update(predictions, labels)
            loss_list.append(loss.item())
        print(f"Test Loss: {np.mean(loss_list)}")
        print(test_metric.__str__())

if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='5', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0 
    args.path_to_trained_model = os.path.join("tested_configs", "resnet18", "initial_config", "model.pt")

    test(args)