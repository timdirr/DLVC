import os
import torch
import torch.nn as nn
from pathlib import Path


class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        if suffix is not None:
            torch.save(self.net.state_dict(), os.path.join(
                save_dir, "model_" + suffix + ".pt"))
        else:
            torch.save(self.net.state_dict(),
                       os.path.join(save_dir, "model.pt"))

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        succ = self.net.load_state_dict(torch.load(path))
        if succ.missing_keys or succ.unexpected_keys:
            raise ValueError(
                "Model architecture does not match the saved model")
