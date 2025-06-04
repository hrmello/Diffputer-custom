import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from generate_mask import generate_mask

DATA_DIR = 'datasets'

def load_dataset(datadir):

    # with open(info_path, 'r') as f:
    #     info = json.load(f)
    
    # data_path = f'{data_dir}/data.csv'
    # train_path = f'{data_dir}/train.csv'
    # test_path = f'{data_dir}/test.csv'
    
    # train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    # test_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'
    
    data_df = pd.read_csv(datadir)
    
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state = 42)

   
    # remove target column
    train_df = train_df.iloc[:,:-1]
    test_df = test_df.iloc[:,:-1]
    
    train_X = train_df.values.astype(np.float32)
    
    return train_X

