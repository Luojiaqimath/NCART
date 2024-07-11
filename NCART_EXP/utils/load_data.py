import numpy as np
import pandas as pd
import os
import pickle


def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q


def load_data(args):
    data_path_list = ["../data/num_and_cat/classification/",
                      "../data/numerical_only/classification/",
                      "../data/num_and_cat/regression/",
                      "../data/numerical_only/regression/"]

    clf_num_cat = os.listdir(data_path_list[0])
    clf_num_only = os.listdir(data_path_list[1])
    reg_num_cat = os.listdir(data_path_list[2])
    reg_num_only = os.listdir(data_path_list[3])
    
    print("Loading dataset " + args.dataset + "...")

    if args.dataset in clf_num_cat:  # num_and_cat feature for clf task
        path = data_path_list[0]
        data = pickle.load(open(path+args.dataset, 'rb'))
        print('data: ' + args.dataset)
        X, y = data[0], data[1]
        
    elif args.dataset in clf_num_only:  # num_only feature for clf task
        path = data_path_list[1]
        data = pickle.load(open(path +args.dataset, 'rb'))
        print('data: ' + args.dataset)
        X, y = data[0], data[1]
        
    elif args.dataset in reg_num_cat:  # num_and_cat feature for reg task
        path = data_path_list[2]
        data = pickle.load(open(path+args.dataset, 'rb'))
        print('data: ' + args.dataset)
        X, y = data[0], data[1]
        
    elif args.dataset in reg_num_only:  # num_only feature for reg task
        path = data_path_list[3]
        data = pickle.load(open(path+args.dataset, 'rb'))
        print('data: ' + args.dataset)
        X, y = data[0], data[1]
        
    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")
    print(X.shape)

    # Setting this if classification task
    if args.objective == "classification":
        args.num_classes = np.max(y)+1
        print("Having", args.num_classes, "classes as target.")

    return X, y
