import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json
from sklearn.model_selection import train_test_split
from generate_mask import generate_mask
from pandas.api.types import is_numeric_dtype
from model import Model, MLPDiffusion

DATA_DIR = 'datasets'

def load_dataset(datadir, info):

    # with open(info_path, 'r') as f:
    #     info = json.load(f)
    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    
    # data_path = f'{data_dir}/data.csv'
    # train_path = f'{data_dir}/train.csv'
    # test_path = f'{data_dir}/test.csv'
    
    # train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    # test_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'
    
    data_df = pd.read_csv(datadir)
    
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state = 42)
    # train_df = train_df.iloc[:,:-1]
    # test_df = test_df.iloc[:,:-1]
    # train_df = pd.read_csv(train_path)
    # test_df = pd.read_csv(test_path)
    print(train_df.values.shape)
    train_mask, test_mask = generate_mask(train_df.values, test_df.values, mask_type = "MCAR", p = 0.3)
    print(train_mask.shape)
    cols = train_df.columns
    
    data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
    data_cat = data_df[cols[cat_col_idx]].astype(str)
    data_y = data_df[cols[target_col_idx]]
    
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]
    
    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = data_cat.columns
    
    train_cat_idx, test_cat_idx = None, None
    extend_train_mask = None
    extend_test_mask = None
    cat_bin_num = None
    
    
    # only contain numerical features
    
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X = test_num
        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask = test_mask[:, num_col_idx]
    
    cols = train_df.columns
    
    #data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
    data_cat = data_df[cols[cat_col_idx]].astype(str)
    data_y = data_df[cols[target_col_idx]]
    
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]
    
    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = data_cat.columns
    target_columns = data_y.columns
    
    train_cat_idx, test_cat_idx = None, None
    
    # Save target idx for target columns
    if len(target_col_idx) != 0 and not is_numeric_dtype(data_y[target_columns[0]]): 
        if not os.path.exists(f'{datadir}/{target_columns[0]}_map_idx.json'):
            print('Creating maps')
            for column in target_columns:
                map_path_bin = f'{datadir}/{column}_map_bin.json'
                map_path_idx = f'{datadir}/{column}_map_idx.json'
                categories = data_y[column].unique()
                num_categories = len(categories) 
    
                num_bits = (num_categories - 1).bit_length()
    
                category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in enumerate(categories)}
                category_to_idx = {category: index for index, category in enumerate(categories)}
                
                # with open(map_path_bin, 'w') as f:
                #     json.dump(category_to_binary, f)
                # with open(map_path_idx, 'w') as f:
                #     json.dump(category_to_idx, f) 
    
        train_target_idx = []
        test_target_idx = []
                
        for column in target_columns:
            map_path_idx = f'{datadir}/{column}_map_idx.json'
            
            # with open(map_path_idx, 'r') as f:
            #     category_to_idx = json.load(f)
                
            train_target_idx_i = train_y[column].map(category_to_idx).to_numpy().astype(np.float32)
            test_target_idx_i = test_y[column].map(category_to_idx).to_numpy().astype(np.float32)
            
            train_target_idx.append(train_target_idx_i)
            test_target_idx.append(test_target_idx_i)
        
        train_target_idx = np.stack(train_target_idx, axis = 1)
        test_target_idx = np.stack(test_target_idx, axis = 1)
    
    else:
        #abuse notation, if the target column is numeric, we still use call it target_idx
        train_target_idx = train_y.to_numpy().astype(np.float32)
        test_target_idx = test_y.to_numpy().astype(np.float32)
    
    # ========================================================
    
    # Save cat idx for cat columns
    if len(cat_col_idx) != 0 and not os.path.exists(f'{data_dir}/{cat_columns[0]}_map_idx.json'):
        print('Creating maps')
        for column in cat_columns:
            map_path_bin = f'{datadir}/{column}_map_bin.json'
            map_path_idx = f'{datadir}/{column}_map_idx.json'
            categories = data_cat[column].unique()
            num_categories = len(categories) 
    
            num_bits = (num_categories - 1).bit_length()
    
            category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in enumerate(categories)}
            category_to_idx = {category: index for index, category in enumerate(categories)}
            
            # with open(map_path_bin, 'w') as f:
            #     json.dump(category_to_binary, f)
            # with open(map_path_idx, 'w') as f:
            #     json.dump(category_to_idx, f)
    
            
    train_cat_idx = []
    test_cat_idx = []
            
    for column in cat_columns:
        map_path_idx = f'{datadir}/{column}_map_idx.json'
        
        # with open(map_path_idx, 'r') as f:
        #     category_to_idx = json.load(f)
            
        train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.float32)
        test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.float32)
        
        train_cat_idx.append(train_cat_idx_i)
        test_cat_idx.append(test_cat_idx_i)
    
    # Four situations:
    # 1. No target columns, no cat columns
    # 2. No target columns, has cat columns
    # 3. Has target columns, no cat columns
    # 4. Has target columns, has cat columns
    if len(target_col_idx) == 0:
    
        if len(cat_col_idx) == 0:
            train_X = train_num
            test_X = test_num
            
            #rearange the column order
            train_X = train_X[:, num_col_idx]
            test_X = test_X[:, num_col_idx]
        else:
            train_cat_idx = np.stack(train_cat_idx, axis = 1)
            test_cat_idx = np.stack(test_cat_idx, axis = 1)
    
            train_X = np.concatenate([train_num, train_cat_idx], axis = 1)
            test_X = np.concatenate([test_num, test_cat_idx], axis = 1)
    
            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx])]
    
    else:
        if len(cat_col_idx) == 0:
            train_X = np.concatenate([train_num, train_target_idx], axis = 1)
            test_X = np.concatenate([test_num, test_target_idx], axis = 1)
    
            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, target_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, target_col_idx])]
            
        else:
            train_cat_idx = np.stack(train_cat_idx, axis = 1)
            test_cat_idx = np.stack(test_cat_idx, axis = 1)
            
            train_X = np.concatenate([train_num, train_cat_idx, train_target_idx], axis = 1)
            test_X = np.concatenate([test_num, test_cat_idx, test_target_idx], axis = 1)
    
            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]
    
    return train_X, test_X, train_mask, test_mask, train_num, test_num, train_cat_idx, test_cat_idx, extend_train_mask, extend_test_mask, cat_bin_num

def recover_num_cat(pred_cat_bin, num_cat, cat_bin_num):
    
    pred_cat_bin[pred_cat_bin <= 0.5] = 0
    pred_cat_bin[pred_cat_bin > 0.5] = 1
    
    pred_cat_bin = pred_cat_bin.astype(np.int32)
    
    cum_sum = cat_bin_num.cumsum()
    cum_sum = np.insert(cum_sum, 0, 0)

    def decode_binary_to_category(binary):
        binary_str = ''.join(map(str, binary))
        index = int(binary_str, 2)
        
        return index
    
    pred_cat = []
    
    for idx in range(num_cat):
        pred_cat_i = pred_cat_bin[:, cum_sum[idx] : cum_sum[idx + 1]]
        pred_cat_i = np.apply_along_axis(decode_binary_to_category, axis=1, arr= pred_cat_i)
        pred_cat.append(pred_cat_i)
    
    pred_cat = np.stack(pred_cat, axis = 1)
    return pred_cat

def load_dataset(datadir, info):

    # with open(info_path, 'r') as f:
    #     info = json.load(f)
    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    
    # data_path = f'{data_dir}/data.csv'
    # train_path = f'{data_dir}/train.csv'
    # test_path = f'{data_dir}/test.csv'
    
    # train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    # test_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'
    
    data_df = pd.read_csv(datadir)
    
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state = 42)
    # train_df = train_df.iloc[:,:-1]
    # test_df = test_df.iloc[:,:-1]
    # train_df = pd.read_csv(train_path)
    # test_df = pd.read_csv(test_path)
    print(train_df.values.shape)
    train_mask, test_mask = generate_mask(train_df.values, test_df.values, mask_type = "MCAR", p = 0.3)
    print(train_mask.shape)
    cols = train_df.columns
    
    data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
    data_cat = data_df[cols[cat_col_idx]].astype(str)
    data_y = data_df[cols[target_col_idx]]
    
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]
    
    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = data_cat.columns
    
    train_cat_idx, test_cat_idx = None, None
    extend_train_mask = None
    extend_test_mask = None
    cat_bin_num = None
    
    
    # only contain numerical features
    
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X = test_num
        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask = test_mask[:, num_col_idx]
    
    cols = train_df.columns
    
    #data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
    data_cat = data_df[cols[cat_col_idx]].astype(str)
    data_y = data_df[cols[target_col_idx]]
    
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]
    
    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = data_cat.columns
    target_columns = data_y.columns
    
    train_cat_idx, test_cat_idx = None, None
    
    # Save target idx for target columns
    if len(target_col_idx) != 0 and not is_numeric_dtype(data_y[target_columns[0]]): 
        if not os.path.exists(f'{datadir}/{target_columns[0]}_map_idx.json'):
            print('Creating maps')
            for column in target_columns:
                map_path_bin = f'{datadir}/{column}_map_bin.json'
                map_path_idx = f'{datadir}/{column}_map_idx.json'
                categories = data_y[column].unique()
                num_categories = len(categories) 
    
                num_bits = (num_categories - 1).bit_length()
    
                category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in enumerate(categories)}
                category_to_idx = {category: index for index, category in enumerate(categories)}
                
                # with open(map_path_bin, 'w') as f:
                #     json.dump(category_to_binary, f)
                # with open(map_path_idx, 'w') as f:
                #     json.dump(category_to_idx, f) 
    
        train_target_idx = []
        test_target_idx = []
                
        for column in target_columns:
            map_path_idx = f'{datadir}/{column}_map_idx.json'
            
            # with open(map_path_idx, 'r') as f:
            #     category_to_idx = json.load(f)
                
            train_target_idx_i = train_y[column].map(category_to_idx).to_numpy().astype(np.float32)
            test_target_idx_i = test_y[column].map(category_to_idx).to_numpy().astype(np.float32)
            
            train_target_idx.append(train_target_idx_i)
            test_target_idx.append(test_target_idx_i)
        
        train_target_idx = np.stack(train_target_idx, axis = 1)
        test_target_idx = np.stack(test_target_idx, axis = 1)
    
    else:
        #abuse notation, if the target column is numeric, we still use call it target_idx
        train_target_idx = train_y.to_numpy().astype(np.float32)
        test_target_idx = test_y.to_numpy().astype(np.float32)
    
    # ========================================================
    
    # Save cat idx for cat columns
    if len(cat_col_idx) != 0 and not os.path.exists(f'{datadir}/{cat_columns[0]}_map_idx.json'):
        print('Creating maps')
        for column in cat_columns:
            map_path_bin = f'{datadir}/{column}_map_bin.json'
            map_path_idx = f'{datadir}/{column}_map_idx.json'
            categories = data_cat[column].unique()
            num_categories = len(categories) 
    
            num_bits = (num_categories - 1).bit_length()
    
            category_to_binary = {category: format(index, '0' + str(num_bits) + 'b') for index, category in enumerate(categories)}
            category_to_idx = {category: index for index, category in enumerate(categories)}
            
            # with open(map_path_bin, 'w') as f:
            #     json.dump(category_to_binary, f)
            # with open(map_path_idx, 'w') as f:
            #     json.dump(category_to_idx, f)
    
            
    train_cat_idx = []
    test_cat_idx = []
            
    for column in cat_columns:
        map_path_idx = f'{datadir}/{column}_map_idx.json'
        
        # with open(map_path_idx, 'r') as f:
        #     category_to_idx = json.load(f)
            
        train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.float32)
        test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.float32)
        
        train_cat_idx.append(train_cat_idx_i)
        test_cat_idx.append(test_cat_idx_i)
    
    # Four situations:
    # 1. No target columns, no cat columns
    # 2. No target columns, has cat columns
    # 3. Has target columns, no cat columns
    # 4. Has target columns, has cat columns
    if len(target_col_idx) == 0:
    
        if len(cat_col_idx) == 0:
            train_X = train_num
            test_X = test_num
            
            #rearange the column order
            train_X = train_X[:, num_col_idx]
            test_X = test_X[:, num_col_idx]
        else:
            train_cat_idx = np.stack(train_cat_idx, axis = 1)
            test_cat_idx = np.stack(test_cat_idx, axis = 1)
    
            train_X = np.concatenate([train_num, train_cat_idx], axis = 1)
            test_X = np.concatenate([test_num, test_cat_idx], axis = 1)
    
            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx])]
    
    else:
        if len(cat_col_idx) == 0:
            train_X = np.concatenate([train_num, train_target_idx], axis = 1)
            test_X = np.concatenate([test_num, test_target_idx], axis = 1)
    
            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, target_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, target_col_idx])]
            
        else:
            train_cat_idx = np.stack(train_cat_idx, axis = 1)
            test_cat_idx = np.stack(test_cat_idx, axis = 1)
            
            train_X = np.concatenate([train_num, train_cat_idx, train_target_idx], axis = 1)
            test_X = np.concatenate([test_num, test_cat_idx, test_target_idx], axis = 1)
    
            #rearange the column order
            train_X = train_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]
            test_X = test_X[:, np.concatenate([num_col_idx, cat_col_idx, target_col_idx])]
    
    return train_X, test_X, train_mask, test_mask, train_num, test_num, train_cat_idx, test_cat_idx, extend_train_mask, extend_test_mask, cat_bin_num



def get_eval(datadir, info, X_recon, X_true, truth_cat_idx, num_num, cat_bin_num, mask):
    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool)

    num_pred = X_recon[:, :num_num]
    cat_pred = X_recon[:, num_num:]

    num_cat = len(cat_col_idx)

    num_true = X_true[:, :num_num]
    cat_true = truth_cat_idx

    # mae = np.abs(num_pred[num_mask] - num_true[num_mask]).mean()
    # rmse = np.sqrt(((num_pred[num_mask] - num_true[num_mask])**2).mean())

    mae = np.nanmean(np.abs(num_pred[num_mask]- num_true[num_mask]))
    rmse = np.sqrt(np.nanmean((num_pred[num_mask]- num_true[num_mask])**2))
    acc = 0

    if num_cat > 0:
        cat_pred = recover_num_cat(cat_pred, num_cat, cat_bin_num)
        acc = (cat_pred[cat_mask] == cat_true[cat_mask]).nonzero()[0].shape[0] / cat_true[cat_mask].shape[0]
        
    return mae, rmse, acc


def imputation(mask: np.array, X: np.array, device: str, hid_dim: int, num_steps: int, ckpt_dir, iteration, rec_Xs: list) -> list:
    X_miss = (1. - mask.float()) * X
    X_miss = X_miss.to(device)
    impute_X = X_miss

    in_dim = X.shape[1]

    denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

    model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)
    model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))

    # ==========================================================

    net = model.denoise_fn_D

    num_samples, dim = X.shape[0], X.shape[1]
    rec_X = impute_mask(net, impute_X, mask, num_samples, dim, num_steps, device)
    
    mask_int = mask.to(torch.float).to(device)
    rec_X = rec_X * mask_int + impute_X * (1-mask_int)
    rec_Xs.append(rec_X)
    print(f'Trial = {trial}')
    return rec_Xs


