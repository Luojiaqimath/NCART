from models.basemodel import BaseModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import torch.nn.functional as F
from utils.io_utils import get_output_path


class BaseModelTorch(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)
        self.device = self.get_device()
        self.gpus = args.gpu_ids if args.use_gpu and torch.cuda.is_available() and args.data_parallel else None

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids).cuda()
            print("On Device: cuda", self.args.gpu_ids)
        else:  
            print("On Device: ", self.device)
            self.model.to(self.device)

    def get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"  
            else:
                device = "cuda:"+str(self.args.gpu_ids[0])
        else:
            device = 'cpu'

        return torch.device(device)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)     
     
    def get_loss_func(self):
        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y_dtype = torch.float32
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
            y_dtype = torch.int64
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y_dtype = torch.float32
        
        return loss_func, y_dtype

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

                out = self.model(batch_X.to(self.device))
    

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

                # Early Stopping
                with torch.no_grad():
                    val_loss = 0.0
                    val_dim = 0
                    for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):

                        out = self.model(batch_val_X.to(self.device))
                

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

    def predict(self, X):
        if self.args.objective == "regression":
            self.predictions = self.predict_helper(X)
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X):
        probas = self.predict_helper(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()
     
        X = np.array(X, dtype=np.float32)
        X = torch.as_tensor(X, dtype=torch.float32)

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size)

        predictions = []

        with torch.no_grad():
            for batch_X in test_loader:
   
                preds = self.model(batch_X[0].to(self.device))
          
                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)
                elif self.args.objective == "classification":
                    preds = F.softmax(preds, dim=1)
                        
                predictions.append(preds.cpu())
        return np.concatenate(predictions)

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")
