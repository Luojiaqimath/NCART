import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from models.basemodel_torch import BaseModelTorch
import numpy as np
import torch.nn.functional as F
import rtdl



class FTTransformer(BaseModelTorch):
    
    def __init__(self, params, args):
        super().__init__(params, args)
        if args.cat_idx:
            self.num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            n_numerical = args.num_features - len(args.cat_idx)
            cat_card = args.cat_dims
        else:
            self.num_idx = list(range(args.num_features))
            n_numerical = args.num_features
            cat_card = None
        self.model = rtdl.FTTransformer.make_baseline(n_num_features=n_numerical,
                            cat_cardinalities=cat_card,
                            d_out=args.num_classes,
                            # d_token=8,
                            n_blocks=self.params["num_blocks"],
                            d_token=self.params["num_token"],
                            ffn_d_hidden=int(2*self.params["num_token"]),
                            attention_dropout=self.params["dropout_att"],
                            ffn_dropout=self.params["dropout_ffn"],
                            residual_dropout=self.params["dropout_res"])
        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None):
        
        loss_func, y_dtype = self.get_loss_func()
        optimizer = self.configure_optimizers()
 

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=y_dtype)
        
        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.float32)
            y_val = torch.as_tensor(y_val, dtype=y_dtype)
        
        train_dataset = TensorDataset(X, y)
        torch.manual_seed(int(torch.sum(y)))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size,
                                  shuffle=True)
        loss_history = []
        
        if X_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            torch.manual_seed(int(torch.sum(y_val)))
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size,
                                    shuffle=True)
            
            min_val_loss = float("inf")
            min_val_loss_idx = 0        
            val_loss_history = []

        for epoch in range(self.args.epochs):
            train_loss = 0.0
            train_dim = 0
            for i, (batch_X, batch_y) in enumerate(train_loader):

                if self.args.cat_idx:
                    x_categ = batch_X[:, self.args.cat_idx].int().to(self.device)
                else:
                    x_categ = None

                x_cont = batch_X[:, self.num_idx].to(self.device)

                out = self.model(x_cont, x_categ)
    
                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                loss = loss_func(out, batch_y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss
                train_dim += 1
            train_loss /= train_dim
            loss_history.append(train_loss.item())
            
            
            if X_val is not None:
                # self.model.eval()
                # Early Stopping
                with torch.no_grad():
                    val_loss = 0.0
                    val_dim = 0
                    for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):

                        if self.args.cat_idx:
                            x_categ = batch_val_X[:, self.args.cat_idx].int().to(self.device)
                        else:
                            x_categ = None

                        x_cont = batch_val_X[:, self.num_idx].to(self.device)

                        out = self.model(x_cont, x_categ)
                

                        if self.args.objective == "regression" or self.args.objective == "binary":
                            out = out.squeeze()

                        val_loss += loss_func(out, batch_val_y.to(self.device))
                        val_dim += 1
                    val_loss /= val_dim
                    val_loss_history.append(val_loss.item())

                    print("Epoch %d: Train Loss %.6f" % (epoch, train_loss), '|', " Val Loss %.6f" % (val_loss))

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        min_val_loss_idx = epoch

                        # Save the currently best model
                        self.save_model(filename_extension="best", directory="tmp")

                    if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                        print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                        print("Early stopping applies.")
                        break
            else:
                print("Epoch %d: Train Loss %.6f" % (epoch, train_loss))
            
                
        self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history

    def predict_helper(self, X):
        self.model.eval()
     
        X = np.array(X, dtype=np.float32)
        X = torch.as_tensor(X, dtype=torch.float32)

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size)

        predictions = []

        with torch.no_grad():
            for batch_X in test_loader:
                x_categ = batch_X[0][:, self.args.cat_idx].int().to(self.device) if self.args.cat_idx else None
                x_cont = batch_X[0][:, self.num_idx].to(self.device)

                preds = self.model(x_cont, x_categ)
                
                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)
                elif self.args.objective == "classification":
                    preds = F.softmax(preds, dim=1)
                        
                predictions.append(preds.cpu())
        return np.concatenate(predictions)
    
    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "num_blocks": trial.suggest_int("num_blocks", 1, 6),
            "num_token": trial.suggest_categorical("num_token", [8, 16, 24, 32]),
            # "num_hidden": trial.suggest_categorical("num_hidden", [4, 6, 8, 10]),
            "dropout_att": trial.suggest_categorical("dropout_att", [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "dropout_ffn": trial.suggest_categorical("dropout_ffn", [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "dropout_res": trial.suggest_categorical("dropout_res", [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        return params