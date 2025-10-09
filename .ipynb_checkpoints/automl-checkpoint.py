import numpy as np
import os

class AutoML:

    def __init__(self):
        pass

    @staticmethod
    def load_dataset(folder_path):

        folder_name = os.path.basename(folder_path)
        
        data = np.loadtxt(f"{folder_path}/{folder_name}.data")
        solution = np.loadtxt(f"{folder_path}/{folder_name}.solution")
        # types = np.loadtxt(f"{folder_path}/{folder_name}.type")
        types = 0

        return data, solution, types

    def fit(self, data_path:str):
        pass

    def eval(self):
        pass