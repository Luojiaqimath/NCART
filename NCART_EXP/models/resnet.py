import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from models.basemodel_torch import BaseModelTorch
import numpy as np
import torch.nn.functional as F
import rtdl



class ResNet(BaseModelTorch):
    
    def __init__(self, params, args):
        super().__init__(params, args)
        torch.manual_seed(args.seed)
        self.model = rtdl.ResNet.make_baseline(d_in=args.num_features,
                            d_out=args.num_classes,
                            d_main=args.num_features,
                            n_blocks=self.params["num_blocks"],
                            d_hidden=self.params["num_units"],
                            dropout_first=self.params["dropout"],
                            dropout_second=self.params["dropout"])
        self.to_device()

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "num_blocks": trial.suggest_int("num_blocks", 1, 10),
            "num_units": trial.suggest_categorical("num_units", [32, 64, 128, 256, 512]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        return params