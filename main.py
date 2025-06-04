import torch
import warnings
import gc
from dataset import load_dataset
from model import DiffPuter
import os
import numpy as np


warnings.filterwarnings('ignore')

def impute_with_column_means(arr):
        arr_copy = arr.copy()
        
        # Calculate mean for each column, ignoring NaN values
        col_means = np.nanmean(arr_copy, axis=0)
        
        # Find NaN positions
        nan_mask = np.isnan(arr_copy)
        
        # Replace NaN values with corresponding column means
        for col in range(arr_copy.shape[1]):
            arr_copy[nan_mask[:, col], col] = col_means[col]
        
        return arr_copy

if __name__ == '__main__':

    # info = {
    # "name": "bean",
    # "header": np.nan,
    # "num_col_idx": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    # "cat_col_idx": [],
    # "target_col_idx": [16]
    # }


    args = {"datadir": "datasets/housing.csv", "dataname": "california", "gpu": "0", "split_idx": 0,
    "max_iter": 10, "ratio": 30, "hid_dim": 1024, "mask": "MCAR", 
        "num_trials": 5, "num_steps": 50, "device": "gpu"}

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    datadir = args["datadir"]
    dataname = args["dataname"]
    split_idx = args["split_idx"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hid_dim = args["hid_dim"]
    mask_type = args["mask"]
    ratio = args["ratio"]
    num_trials = args["num_trials"]
    num_steps = args["num_steps"]
 
    print(torch.cuda.is_available())
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    train_X= load_dataset(datadir = datadir)

    train_X = impute_with_column_means(train_X)

    # remove target column
    train_X = train_X[:,:-1]
    
    mean_X = train_X.mean(0)
    std_X = train_X.std(0)
    in_dim = train_X.shape[1]

    result_save_path = f'results/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
    os.makedirs(result_save_path) if not os.path.exists(result_save_path) else None
    
    diffputer = DiffPuter(result_save_path = result_save_path,
                          num_trials = 10, 
                          epochs_m_step = 10000, 
                          patience_m_step = 300, 
                          batch_size = 8192,
                          hid_dim = 1024, 
                          device = "cuda", 
                          max_iter = 10, 
                          lr = 1e-4, 
                          num_steps = 50, 
                          ckpt_dir = "/home/kunumi/Área de trabalho/Diffputer-custom")
    
    diffputer.fit(train_X = train_X)
    gc.collect()