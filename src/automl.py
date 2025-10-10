import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier

class AutoML:

    def __init__(self):
        self.models = {
            'regression': [
                ('Linear Regression', LinearRegression()),
                ('Random Forest Regressor', RandomForestRegressor(random_state=42)),
                ('K-Neighbors Regressor', KNeighborsRegressor())
            ],
            'binary_classification': [
                ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
                ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
                ('K-Neighbors Classifier', KNeighborsClassifier())
            ],
            'multiclass_classification': [
                ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
                ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
                ('K-Neighbors Classifier', KNeighborsClassifier())
            ],
            'multilabel_classification': [
                ('Random Forest Multi-label', MultiOutputClassifier(RandomForestClassifier(random_state=42))),
                ('K-Neighbors Multi-label', MultiOutputClassifier(KNeighborsClassifier()))
            ]
        }

    @staticmethod
    def load_dataset(folder_path: str):
        basename = os.path.basename(folder_path)
        data = pd.read_csv(f"{folder_path}/{basename}.data", sep=" ", header=None)
        data.columns = [f'feature_{i}' for i in range(data.shape[1])]
        
        solution = np.loadtxt(f"{folder_path}/{basename}.solution")
        
        with open(f"{folder_path}/{basename}.type", "r", encoding="utf-8") as f:
            ntypes = [line.strip() for line in f.readlines()]
        types = np.array(ntypes)      
        return data, solution, types


    @staticmethod
    def detect_task_type(solution:np.array) -> str:
        """Trouve le type de probleme (classification, regression etc)"""

        if solution.ndim > 1:
            # Compter le nombre de 1 par ligne
            count = np.sum(np.sum(solution == 1, axis=1))
            
            if count == solution.shape[0]:
                # 0 0 0 1
                # 0 1 0 0
                return "multiclass_classification"
            else:
                # 1 1 0 1
                # 0 1 1 0
                return "multilabel_classification"
            
        else:
            sflat = solution.flatten()
            unique_values = np.unique(sflat)
            
            if len(unique_values) == 2:
                # 0
                # 1
                return "binary_classification"
            else:
                # 123
                # 241
                return "regression"
            
        return None

        

    def fit(self, folder:str):

        # Charger les données
        data, solution, types = self.load_dataset(folder)
        
        # Trouver le type de probleme
        problem_type = self.detect_task_type(solution)
        if problem_type is None:
            raise ValueError(f"Le problème n'a pas été identifié !")
        print(f"Probleme de type : {problem_type}")

        # Tester les models qui correspondent
        for model in self.models[problem_type]:
            print(f"Test du model : {model[0]}")


    

    def eval(self):
        pass






        