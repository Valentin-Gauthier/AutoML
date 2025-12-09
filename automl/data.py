# Fichier: automl/data_loader.py
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os

def load_sparce_matrix(data_path:str):
    """Charger une matrice creuse"""
    rows = []
    cols = []
    data = []

    with open(data_path, 'r') as f:
        for row_idx, line in enumerate(f):
            elements = line.strip().split()

            for item in elements:
                idx_str, val_str = item.split(":")
                col_idx = int(idx_str)
                val = float(val_str)

                rows.append(row_idx)
                cols.append(col_idx)
                data.append(val)
                
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    sparse_mat = coo_matrix((data, (rows, cols)))

    return sparse_mat

def load_dataset(folder_path: str):
    basename = os.path.basename(folder_path)
    # Protection si le chemin finit par un slash
    if not basename:
        basename = os.path.basename(folder_path[:-1])
        
    data = pd.read_csv(f"{folder_path}/{basename}.data", sep=r"\s+", header=None, na_values='NaN', engine='python')
    data.columns = [f'feature_{i}' for i in range(data.shape[1])]
    
    solution = np.loadtxt(f"{folder_path}/{basename}.solution")
    
    with open(f"{folder_path}/{basename}.type", "r", encoding="utf-8") as f:
        ntypes = [line.strip() for line in f.readlines()]
    types = np.array(ntypes)      
    return data, solution, types