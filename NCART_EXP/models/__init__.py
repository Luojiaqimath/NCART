all_models = ["XGBoost", "CatBoost", "LightGBM", "TabNet", "NODE", "ResNet", "NCART", "MLP", "SAINT", "RLN"]


def str2model(model):
    if model == "XGBoost":
        from models.tree_models import XGBoost
        return XGBoost

    elif model == "CatBoost":
        from models.tree_models import CatBoost
        return CatBoost
    
    elif model == "LightGBM":
        from models.tree_models import LightGBM
        return LightGBM

    elif model == "TabNet":
        from models.tabnet import TabNet
        return TabNet

    elif model == "NODE":
        from models.node import NODE
        return NODE

    elif model == "ResNet":
        from models.resnet import ResNet
        return ResNet
    
    elif model == "FTTransformer":
        from models.fttransformer import FTTransformer
        return FTTransformer
    
    elif model == "MLP":
        from models.mlp import MLP
        return MLP
    
    elif model == "SAINT":
        from models.saint import SAINT
        return SAINT
    
    elif model == "RLN":
        from models.rln import RLN
        return RLN
    
    elif model == "NCART":
        from models.ncart import NCART
        return NCART

    else:
        raise NotImplementedError("Model \"" + model + "\" not yet implemented")
