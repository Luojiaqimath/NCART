import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from models.basemodel_torch import BaseModelTorch
from entmax import sparsemax, entmax15
import numpy as np


class NCART(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = DeepCART(args.num_features,
                              self.params["n_selected"],
                              args.num_classes,
                              n_trees=self.params["n_trees"],
                              n_layers=self.params["n_layers"],
                              mask_type=self.params["mask_type"])
        self.to_device()

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_trees": trial.suggest_categorical("n_trees", [8, 16, 32, 64]),
            "n_selected": trial.suggest_int("n_selected", 2, 10),
            "n_layers": trial.suggest_categorical("n_layers", [2, 4]),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        }
        return params


class CART(nn.Module):
    def __init__(self, n_features, n_selected, n_out, n_trees, mask_type):
        super(CART, self).__init__()
        self.n_features = n_features
        self.n_selected = n_selected
        self.n_trees = n_trees
        self.n_out = n_out
        self.mask_type = mask_type
        self.bn = nn.BatchNorm1d(n_features)

        if n_features > n_selected and n_features > n_out:
            self.projection = True
            self.feature_selection_matric = nn.Parameter(torch.randn((n_trees,
                                                                      n_features,
                                                                      n_selected),
                                                                     requires_grad=True))
            self.cut_points = nn.Parameter(torch.randn(
                (n_trees, 1, n_selected), requires_grad=True))
            self.linear_layer1 = nn.Parameter(torch.randn(
                (n_trees, n_selected, n_selected), requires_grad=True))
            self.bias1 = nn.Parameter(torch.randn(
                (n_trees, 1, n_selected), requires_grad=True))
            self.linear_layer2 = nn.Parameter(torch.randn(
                (n_trees, n_selected, n_out), requires_grad=True))
            self.bias2 = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))
            self.tree_weight = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))

        else:
            self.projection = False
            self.cut_points = nn.Parameter(torch.zeros(
                (n_trees, 1, n_features), requires_grad=True))
            self.linear_layer1 = nn.Parameter(torch.randn(
                (n_trees, n_features, n_features), requires_grad=True))
            self.bias1 = nn.Parameter(torch.randn(
                (n_trees, 1, n_features), requires_grad=True))
            self.linear_layer2 = nn.Parameter(torch.randn(
                (n_trees, n_features, n_out), requires_grad=True))
            self.bias2 = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))
            self.tree_weight = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))

    def forward(self, x):
        # N:number of trees, B:batch size, D:dim of features, S:dim of selected features

        x = self.bn(x)  # (B, D) -> (B, D)

        if self.projection:  # (N, B, D)*(D, S) -> (N, B, S)
            if self.mask_type == 'sparsemax':
                projection_matrix = sparsemax(
                    self.feature_selection_matric, dim=1)
                x = torch.matmul(x, projection_matrix)
            elif self.mask_type == 'entmax':
                projection_matrix = entmax15(
                    self.feature_selection_matric, dim=1)
                x = torch.matmul(x, projection_matrix)
            else:
                raise NotImplementedError("Method not yet implemented")

        diff = x-self.cut_points
        score = torch.sigmoid(diff)

        # first linear layer for n trees, (N, B, D)*(N, D, D) -> (N, B, D),  
        # Z_nbo = sum_k(X_nbk*Y_nko)
        o1 = F.relu(torch.einsum('nbk, nko->nbo', score,
                    self.linear_layer1) + self.bias1)

        # second linear layer for n trees, (N, B, D)*(N, D, O) -> (N, B, O)
        o2 = torch.einsum('nbk, nko->nbo', o1, self.linear_layer2) + \
            self.bias2  # Z_nbo = sum_k(X_nbk*Y_nko)

        # merge all trees, (N, B, O)*(N, 1, O) -> (B, O)
        o3 = (o2*self.tree_weight).mean(0)
        return o3


class DeepCART(nn.Module):
    def __init__(self, n_features, n_selected, n_out, n_trees, n_layers, mask_type):
        super(DeepCART, self).__init__()
        self.n_layers = n_layers
        if self.n_layers > 1:
            self.model_list = nn.ModuleList([CART(n_features,
                                                  n_selected,
                                                  n_features,
                                                  n_trees,
                                                  mask_type) for _ in range(n_layers-1)])
        self.last_layer = CART(n_features,
                               n_selected,
                               n_out,
                               n_trees,
                               mask_type)

    def forward(self, x):
        if self.n_layers > 1:
            for m in self.model_list:
                x = F.relu(m(x) + x)
        out = self.last_layer(x)
        return out



# class NCART(BaseModelTorch):

#     def __init__(self, params, args):
#         super().__init__(params, args)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
#         self.model = DeepCART(args.num_features,
#                               self.params["n_selected"],
#                               args.num_classes,
#                               n_trees=self.params["n_trees"],
#                               n_layers=self.params["n_layers"],
#                               mask_type=self.params["mask_type"])
#         self.to_device()

#     @classmethod
#     def define_trial_parameters(cls, trial, args):
#         params = {
#             "n_trees": trial.suggest_categorical("n_trees", [8, 16, 32, 64]),
#             "n_selected": trial.suggest_int("n_selected", 2, 10),
#             "n_layers": trial.suggest_categorical("n_layers", [2, 4]),
#             "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
#         }
#         return params


# class CART(nn.Module):
#     def __init__(self, n_features, n_selected, n_out, n_trees, mask_type):
#         super(CART, self).__init__()
#         self.n_features = n_features
#         self.n_selected = n_selected
#         self.n_trees = n_trees
#         self.n_out = n_out
#         self.mask_type = mask_type
#         self.bn = nn.BatchNorm1d(n_features)

#         if n_features > n_selected and n_features > n_out:
#             self.projection = True
#             self.feature_selection_matrix = nn.Parameter(torch.randn((n_trees,
#                                                                       n_features,
#                                                                       n_selected),
#                                                                      requires_grad=True))
#             self.cut_points = nn.Parameter(torch.randn(
#                 (n_trees, 1, n_selected), requires_grad=True))
#             self.linear_layer1 = nn.Parameter(torch.randn(
#                 (n_trees, n_selected, n_selected), requires_grad=True))
#             self.bias1 = nn.Parameter(torch.randn(
#                 (n_trees, 1, n_selected), requires_grad=True))
#             self.linear_layer2 = nn.Parameter(torch.randn(
#                 (n_trees, n_selected, n_out), requires_grad=True))
#             self.bias2 = nn.Parameter(torch.randn(
#                 (n_trees, 1, n_out), requires_grad=True))
#             self.tree_weight = nn.Parameter(torch.randn(
#                 (n_trees, 1, n_out), requires_grad=True))

#         else:
#             self.projection = False
#             self.cut_points = nn.Parameter(torch.zeros(
#                 (n_trees, 1, n_features), requires_grad=True))
#             self.linear_layer1 = nn.Parameter(torch.randn(
#                 (n_trees, n_features, n_features), requires_grad=True))
#             self.bias1 = nn.Parameter(torch.randn(
#                 (n_trees, 1, n_features), requires_grad=True))
#             self.linear_layer2 = nn.Parameter(torch.randn(
#                 (n_trees, n_features, n_out), requires_grad=True))
#             self.bias2 = nn.Parameter(torch.randn(
#                 (n_trees, 1, n_out), requires_grad=True))
#             self.tree_weight = nn.Parameter(torch.randn(
#                 (n_trees, 1, n_out), requires_grad=True))

#     def forward(self, x):
#         # N:number of trees, B:batch size, D:dim of features, S:dim of selected features

#         x = self.bn(x)  # (B, D) -> (B, D)

#         if self.projection:  # (B, D)*(N, D, S) -> (N, B, S)
#             if self.mask_type == 'sparsemax':
#                 projection_matrix = sparsemax(
#                     self.feature_selection_matrix, dim=1)
#                 x = torch.matmul(x, projection_matrix)
#             elif self.mask_type == 'entmax':
#                 projection_matrix = entmax15(
#                     self.feature_selection_matrix, dim=1)
#                 x = torch.matmul(x, projection_matrix)
#             else:
#                 raise NotImplementedError("Method not yet implemented")

#         diff = x-self.cut_points
#         score = torch.sigmoid(diff)

#         # first linear layer for n trees, (N, B, D)*(N, D, D) -> (N, B, D)
#         # Z_nbo = sum_k(X_nbk*Y_nko)
#         o1 = F.relu(torch.einsum('nbk, nko->nbo', score,
#                     self.linear_layer1) + self.bias1)

#         # second linear layer for n trees, (N, B, D)*(N, D, O) -> (N, B, O)
#         o2 = torch.einsum('nbk, nko->nbo', o1, self.linear_layer2) + \
#             self.bias2  # Z_nbo = sum_k(X_nbk*Y_nko)

#         # merge all trees, (N, B, O)*(N, 1, O) -> (B, O)
#         o3 = (o2*self.tree_weight).mean(0)
#         return o3


# class DeepCART(nn.Module):
#     def __init__(self, n_features, n_selected, n_out, n_trees, n_layers, mask_type):
#         super(DeepCART, self).__init__()
#         self.n_layers = n_layers
#         if self.n_layers > 1:
#             self.model_list = nn.ModuleList([CART(n_features,
#                                                   n_selected,
#                                                   n_features,
#                                                   n_trees,
#                                                   mask_type) for _ in range(n_layers-1)])
#         self.last_layer = CART(n_features,
#                                n_selected,
#                                n_out,
#                                n_trees,
#                                mask_type)

#     def forward(self, x):
#         if self.n_layers > 1:
#             for m in self.model_list:
#                 x = F.relu(m(x) + x)
#         out = self.last_layer(x)
#         return out
