import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15
import numpy as np

    
class NCARClassifier():

    def __init__(self, 
                n_trees=32,
                n_selected=8,
                n_layers=4,
                mask_type='sparsemax',
                batch_size=1024,
                val_batch_size=256,
                epochs=1000,
                early_stop_round=10,
                use_gpu=True,
                gpu_ids=[0],
                data_parallel=False,
                seed=2023):
        super().__init__()
        
        self.n_trees = n_trees
        self.n_selected = n_selected
        self.n_layers = n_layers
        self.mask_type = mask_type
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.early_stop_round = early_stop_round
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.data_parallel = data_parallel 
        self.device = self.get_device(use_gpu, gpu_ids, data_parallel)
        self.gpus = gpu_ids if use_gpu and torch.cuda.is_available() and data_parallel else None
        self.seed = seed

    def to_device(self):
        if self.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids).cuda()
            print("On Device: cuda", self.gpu_ids)
        else:  
            print("On Device: ", self.device)
            self.model.to(self.device)

    def get_device(self, use_gpu, gpu_ids, data_parallel):
        if use_gpu and torch.cuda.is_available():
            if data_parallel:
                device = "cuda"  
            else:
                device = "cuda:"+str(gpu_ids[0])
        else:
            device = 'cpu'

        return torch.device(device)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)     
     
    def get_loss_func(self):
        if self.n_out > 1:
            print('Multi-class classification')
            loss_func = nn.CrossEntropyLoss()
            y_dtype = torch.int64
        else:
            print('Binary classification')
            loss_func = nn.BCEWithLogitsLoss()
            y_dtype = torch.float32
        
        return loss_func, y_dtype

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        self.n_out = 1 if np.max(y)==1 else np.max(y)+1  # n_out=1, binary, n_out>1, classification
        self.n_features = X.shape[1]
        self.model = NCART(self.n_features,
                              self.n_selected,
                              self.n_out,
                              n_trees=self.n_trees,
                              n_layers=self.n_layers,
                              mask_type=self.mask_type)
        self.to_device()
        
        loss_func, y_dtype = self.get_loss_func()
        optimizer = self.configure_optimizers()
 
        X = torch.as_tensor(X, dtype=torch.float32)
        self.X = X.to(self.device)
        y = torch.as_tensor(y, dtype=y_dtype)
        
        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.float32)
            y_val = torch.as_tensor(y_val, dtype=y_dtype)
        
        train_dataset = TensorDataset(X, y)
        torch.manual_seed(int(torch.sum(y)))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                  shuffle=True)
        loss_history = []
        val_loss_history = []
        
        if X_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            torch.manual_seed(int(torch.sum(y_val)))
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.val_batch_size,
                                    shuffle=True)
            
            min_val_loss = float("inf")
            min_val_loss_idx = 0        
            

        for epoch in range(self.epochs):
            train_loss = 0.0
            train_dim = 0
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.model(batch_X.to(self.device))
    
                if self.n_out==1:
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
                
                        if self.n_out==1:
                            out = out.squeeze()

                        val_loss += loss_func(out, batch_val_y.to(self.device))
                        val_dim += 1
                    val_loss /= val_dim
                    val_loss_history.append(val_loss.item())

                    print("Epoch %d: Train Loss %.6f" % (epoch, train_loss), '|', " Val Loss %.6f" % (val_loss))

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        min_val_loss_idx = epoch

                    if min_val_loss_idx + self.early_stopping_round < epoch:
                        print("Validation loss has not improved for %d steps!" % self.early_stopping_round)
                        print("Early stopping applies.")
                        break
            else:
                print("Epoch %d: Train Loss %.6f" % (epoch, train_loss))
            
                
        return loss_history, val_loss_history

    def predict(self, X):
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

    def get_importance(self):
        if self.data_parallel:
            return self.model.module.feature_importance(self.X)
        else:
            return self.model.feature_importance(self.X)   
    
    
class NCARRegressor():

    def __init__(self, 
                n_trees=32,
                n_selected=8,
                n_layers=4,
                mask_type='sparsemax',
                batch_size=1024,
                val_batch_size=256,
                epochs=1000,
                early_stop_round=10,
                use_gpu=True,
                gpu_ids=[0],
                data_parallel=False,
                seed=2023):
        super().__init__()
        
        self.n_trees = n_trees
        self.n_selected = n_selected
        self.n_layers = n_layers
        self.mask_type = mask_type
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.early_stop_round = early_stop_round
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.data_parallel = data_parallel 
        self.device = self.get_device(use_gpu, gpu_ids, data_parallel)
        self.gpus = gpu_ids if use_gpu and torch.cuda.is_available() and data_parallel else None
        self.seed = seed

    def to_device(self):
        if self.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids).cuda()
            print("On Device: cuda", self.gpu_ids)
        else:  
            print("On Device: ", self.device)
            self.model.to(self.device)

    def get_device(self, use_gpu, gpu_ids, data_parallel):
        if use_gpu and torch.cuda.is_available():
            if data_parallel:
                device = "cuda"  
            else:
                device = "cuda:"+str(gpu_ids[0])
        else:
            device = 'cpu'

        return torch.device(device)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)     
     
    def get_loss_func(self):
        loss_func = nn.MSELoss()
        y_dtype = torch.float32
        
        return loss_func, y_dtype

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        self.n_out = 1 
        self.n_features = X.shape[1]
        self.model = NCART(self.n_features,
                              self.n_selected,
                              self.n_out,
                              n_trees=self.n_trees,
                              n_layers=self.n_layers,
                              mask_type=self.mask_type)
        self.to_device()
        
        loss_func, y_dtype = self.get_loss_func()
        optimizer = self.configure_optimizers()
 
        X = torch.as_tensor(X, dtype=torch.float32)
        self.X = X.to(self.device)
        y = torch.as_tensor(y, dtype=y_dtype)
        
        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.float32)
            y_val = torch.as_tensor(y_val, dtype=y_dtype)
        
        train_dataset = TensorDataset(X, y)
        torch.manual_seed(int(torch.sum(y)))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                  shuffle=True)
        loss_history = []
        val_loss_history = []
        
        if X_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            torch.manual_seed(int(torch.sum(y_val)))
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.val_batch_size,
                                    shuffle=True)
            
            min_val_loss = float("inf")
            min_val_loss_idx = 0        
            

        for epoch in range(self.epochs):
            train_loss = 0.0
            train_dim = 0
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.model(batch_X.to(self.device))
    
                if self.n_out==1:
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
                
                        if self.n_out==1:
                            out = out.squeeze()

                        val_loss += loss_func(out, batch_val_y.to(self.device))
                        val_dim += 1
                    val_loss /= val_dim
                    val_loss_history.append(val_loss.item())

                    print("Epoch %d: Train Loss %.6f" % (epoch, train_loss), '|', " Val Loss %.6f" % (val_loss))

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        min_val_loss_idx = epoch

                    if min_val_loss_idx + self.early_stopping_round < epoch:
                        print("Validation loss has not improved for %d steps!" % self.early_stopping_round)
                        print("Early stopping applies.")
                        break
            else:
                print("Epoch %d: Train Loss %.6f" % (epoch, train_loss))
            
                
        return loss_history, val_loss_history

    def predict(self, X):
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

    def get_importance(self):
        if self.data_parallel:
            return self.model.module.feature_importance(self.X)
        else:
            return self.model.feature_importance(self.X)
    
    


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
            self.feature_selection_matrix = nn.Parameter(torch.randn((n_trees,
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
            self.cut_points = nn.Parameter(torch.randn(
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

        if self.projection:  # (B, D)*(N, D, S) -> (N, B, S)
            if self.mask_type == 'sparsemax':
                self.projection_matrix = sparsemax(
                    self.feature_selection_matrix, dim=1)
                x = torch.matmul(x, self.projection_matrix)
            elif self.mask_type == 'entmax':
                self.projection_matrix = entmax15(
                    self.feature_selection_matrix, dim=1)
                x = torch.matmul(x, self.projection_matrix)
            else:
                raise NotImplementedError("Method not yet implemented")

        diff = x-self.cut_points
        score = torch.sigmoid(diff)

        # first linear layer for n trees, (N, B, D)*(N, D, D) -> (N, B, D)
        # Z_nbo = sum_k(X_nbk*Y_nko)
        o1 = F.relu(torch.einsum('nbk, nko->nbo', score,
                    self.linear_layer1) + self.bias1)

        # second linear layer for n trees, (N, B, D)*(N, D, O) -> (N, B, O)
        # Z_nbo = sum_k(X_nbk*Y_nko)
        o2 = torch.einsum('nbk, nko->nbo', o1, self.linear_layer2) + self.bias2 

        # merge all trees, (N, B, O)*(N, 1, O) -> (B, O)
        o3 = (o2*self.tree_weight).mean(0)
        return o3
    
    def get_index(self, x):  
        x = self.bn(x)
        if self.projection:
            x = torch.matmul(x, self.projection_matrix)
        return torch.heaviside((x-self.cut_points).detach().cpu(), torch.tensor([0.0]))
        # shape = (N, A, D) or (N, A, S), where A is the input data number
    
    def get_gini(self, x):
        num_data = x.shape[0]
        index = self.get_index(x)  # (N, A, D) or (N, A, S)
        if self.projection:
            pos = index.sum(-2, keepdim=True)/num_data   # (N, 1, S)
            neg = 1-pos
            return ((1.-pos**2-neg**2)*self.projection_matrix.detach().cpu()).sum(-1).sum(0)  
            # (N, 1, S)*(N, D, S) -> (N, D, S), -> (N, D) -> (N)
        else:
            pos = index.sum(-2)/num_data   # (N, D)
            neg = 1-pos
            return (1.-pos**2-neg**2).sum(0)  # (D)      


class NCART(nn.Module):
    def __init__(self, n_features, n_selected, n_out, n_trees, n_layers, mask_type):
        super(NCART, self).__init__()
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.n_features = n_features
        self.n_selected = n_selected
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
    
    def feature_importance(self, x):
        importance = torch.zeros(x.shape[1])
        if self.n_layers > 1:
            for m in self.model_list:
                importance += m.get_gini(x)
        importance += self.last_layer.get_gini(x)
        return self.normalize(importance.numpy())
    
    def normalize(self, x):
        return x/np.sum(x)
        
