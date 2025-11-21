import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os
import tqdm
import yaml
from .models import MODELS_CONFIG

# Normaliser des données
from sklearn.preprocessing import StandardScaler
# Remplacer les valeurs manquantes
from sklearn.impute import SimpleImputer
# One Hot Encoder poour les données 
from sklearn.preprocessing import OneHotEncoder
# division des données en données d'entrainement et de tests
from sklearn.model_selection import train_test_split
# Evaluer les models
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, make_scorer
# Optimiser les hyperparametres
from sklearn.model_selection import GridSearchCV

class AutoML:

    def __init__(self):
        # models
        self.models = MODELS_CONFIG
        # config des hyperparametres
        with open("src/models_config.yaml", 'r') as f:
            self.param_grids = yaml.safe_load(f)
        
        self.trained_models = {}
        self.scores = {}
        self.task_type = None
        self.X_test = None
        self.y_test = None
        self.best_model = None
        self.best_params = {}

    @staticmethod
    def load_sparce_matrix(data_path:str):
        """Charger une matrice creuse"""
        rows = []
        cols = []
        data = []

        with open(data_path, 'r') as f:
            for row_idx, line in enumerate(f):
                elements = line.strip().split()

                for item in elements: # 24:1
                    idx_str, val_str = item.split(":")
                    col_idx = int(idx_str)
                    val = float(val_str)

                    rows.append(row_index)
                    cols.append(col_idx)
                    data.append(val)
                    
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)

        sparse_mat = coo_matrix((data, (rows, cols)))

        return sparse_mat
                    
        
        
    
    @staticmethod
    def load_dataset(folder_path: str):
        basename = os.path.basename(folder_path)
        data = pd.read_csv(f"{folder_path}/{basename}.data", sep=r"\s+", header=None, na_values='NaN', engine='python')
        data.columns = [f'feature_{i}' for i in range(data.shape[1])]
        
        solution = np.loadtxt(f"{folder_path}/{basename}.solution")
        
        with open(f"{folder_path}/{basename}.type", "r", encoding="utf-8") as f:
            ntypes = [line.strip() for line in f.readlines()]
        types = np.array(ntypes)      
        return data, solution, types

    def get_scoring_metric(self, task_type: str):
        if task_type == "binary_classification":
            return "roc_auc" # ou 'accuracy'
        elif task_type == "multiclass_classification":
            return "f1_weighted" # Mieux que accuracy si classes déséquilibrées
        elif task_type == "multilabel_classification":
            return make_scorer(f1_score, average='samples', zero_division=0)
        elif task_type == "regression":
            return "r2" # ou 'neg_root_mean_squared_error'
        else:
            return "accuracy"

    @staticmethod
    def detect_task_type(solution:np.array) -> str:
        """Trouve le type de probleme (classification, regression etc)"""
        # Si il y a plus de 1 dimensions
        if solution.ndim > 1:
            # Compter le nombre de 1 par ligne
            # Si un seul 1 par ligne alors multiclasse sinon multi-labels
            if np.all(np.sum(solution, axis=1) == 1):
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

    def fit(self, folder:str, test_size:float=0.2, verbose:bool=False):
        """
        Charger les données, trouver le type de problème, preparer les données et tester les différents modèle de sklearn associé.
        
        Args:
            - folder (str): Chemin vers le dossier du dataset.
            - test_size (float): Taille du batch de test ]0, 1[
            - verbose (bool): Active certain log
        """
        # 1) Charger le dataset
        data, solution, types = self.load_dataset(folder)
        
        # 2) identifier le type de problème (classification/regression etc)
        self.task_type = self.detect_task_type(solution)
        if verbose:
            print(f"-> Tache de type {self.task_type} detecté.")

        # 3) Preparation des données
        # - a) separation des données en train/test
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            data, solution, test_size=test_size, random_state=42
        )
        # - b) Recuperer le type de données associé a chaque colonne
        cat_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Categorical"]
        num_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Numerical"]
        bin_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Binary"]
        # - c) traiter chaque features en fonction de leur type

        # Données de type numerique
        if num_cols:
            # 1) imputation
            num_imputer = SimpleImputer(strategy="median")
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            self.X_test[num_cols] = num_imputer.transform(self.X_test[num_cols])
            # 2) Normalisation
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            self.X_test[num_cols] = scaler.transform(self.X_test[num_cols])

        # Données de type binaire
        if bin_cols:
            # 1) imputation
            bin_imputer = SimpleImputer(strategy="most_frequent")
            X_train[bin_cols] = bin_imputer.fit_transform(X_train[bin_cols])
            self.X_test[bin_cols] = bin_imputer.transform(self.X_test[bin_cols])

        # Données de type catégorielle
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            self.X_test[cat_cols] = cat_imputer.transform(self.X_test[cat_cols])
        
            # Encodage One-Hot
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            # On apprend les catégories sur X_train et on le transforme
            X_train_encoded = ohe.fit_transform(X_train[cat_cols])
            X_test_encoded = ohe.transform(self.X_test[cat_cols])

            # On récupère les nouveaux noms de colonnes créés par l'encodeur
            encoded_cols = ohe.get_feature_names_out(cat_cols)
            # On crée des DataFrames avec les données encodées et les bons noms de colonnes
            X_train_encoded_df = pd.DataFrame(X_train_encoded, index=X_train.index, columns=encoded_cols)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, index=self.X_test.index, columns=encoded_cols)
        
            # On supprime les colonnes catégorielles originales
            X_train = X_train.drop(cat_cols, axis=1)
            self.X_test = self.X_test.drop(cat_cols, axis=1)
        
            # On fusionne les DataFrames originaux avec les nouvelles colonnes encodées
            X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
            self.X_test = pd.concat([self.X_test, X_test_encoded_df], axis=1)

        # 4) Analyse des données ???

        # 5) Entrainement des models correspondant
        models_to_test = self.models[self.task_type]
        if verbose:
            print("\n-> Choix des models pertinant : \n")
            models_name = [name for name, model in models_to_test]
            for i in range(len(models_name)):
                print(f"{i+1} - {models_name[i]}")

        for model_name, model in models_to_test:
            try:
                if verbose:
                    print(f"\n Optimisation du modèle : {model_name}")
                if model_name in self.param_grids:

                    # Créer l'objet GridSearchCV
                    # cv=3 signifie 3-fold cross-validation
                    # n_jobs=-1 utilise tous les cœurs CPU disponibles sur le cluster Skinner
                    grid_search = GridSearchCV(
                        model, 
                        self.param_grids[model_name], 
                        cv=3, 
                        scoring=self.get_scoring_metric(self.task_type), 
                        n_jobs=-1
                    )
                    
                    # Lancer la recherche
                    grid_search.fit(X_train, y_train)
                    print(f"   => Meilleurs paramètres trouvés : {grid_search.best_params_}")
                    # Le meilleur modèle est maintenant l'attribut best_estimator_
                    self.trained_models[model_name] = grid_search.best_estimator_
                    self.best_params[model_name] = grid_search.best_params_
                else:
                    # Si pas de grille définie, on fait un entraînement simple
                    model.fit(X_train, y_train)
                    self.trained_models[model_name] = model
                    self.best_params[model_name] = "N/A (Paramètres par défaut)"
                
            except Exception as e:
                print(f"Erreur avec le modèle {model_name} : {e}")
    


    def eval(self):
        if not self.trained_models:
            print(f"Aucun model n'a ete entrainé")
            return None

        print("\n -> Résultats : \n")
        for model_name, model in self.trained_models.items():
            # Prédiction sur les données de test
            y_pred = model.predict(self.X_test)
            
            score = 0
            metric = ""
            # Choisir une meilleure métrique pour chaque type de problème
            if self.task_type == "regression":
                score = mean_squared_error(self.y_test, y_pred)
                metric = "MSE"
            elif self.task_type == "multilabel_classification":
                # Ajoutez zero_division=0.0 pour dire à la fonction de mettre 0 sans afficher d'avertissement.
                score = f1_score(self.y_test, y_pred, average='samples', zero_division=0.0)
                metric = "F1-Score (samples)"
            else: # Binaire ou Multi-classe
                score = accuracy_score(self.y_test, y_pred)
                metric = "Accuracy"
            
            self.scores[model_name] = score
            print(f"Model: {model_name} et Score {metric}: {score:.4f}")

        # Trouver le meilleur model
        if self.scores:
            if self.task_type == "regression":
                # Pour la régression, on cherche le score le plus bas 
                best_model_name = min(self.scores, key=self.scores.get)
            else:
                # Pour la classification, on cherche le score le plus haut
                best_model_name = max(self.scores, key=self.scores.get)
            
            self.best_model = self.trained_models[best_model_name]
            best_model_score = self.scores[best_model_name]
            best_model_params = self.best_params[best_model_name]
            
            # --- Votre nouveau bloc de récapitulatif ---
            print(f"\n------------------------------------------------")
            print(f" RÉCAPITULATIF DU MEILLEUR MODÈLE ")
            print(f"------------------------------------------------")
            print(f"Modèle         : {best_model_name}")
            print(f"Score ({metric}): {best_model_score:.4f}")
            print(f"Hyperparamètres : {best_model_params}")
            print(f"------------------------------------------------")



        