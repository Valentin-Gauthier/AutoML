import pandas as pd
import numpy as np
import os
# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier

# Normaliser des données
from sklearn.preprocessing import StandardScaler
# Remplacer les valeurs manquantes
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
# Evaluer les models
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

# Optimiser les hyperparametres
from sklearn.model_selection import GridSearchCV

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

        self.param_grids = {
            # Grilles pour la Régression 
            'Random Forest Regressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'max_features': ['sqrt', 'log2', 1.0]
            },
            'K-Neighbors Regressor': {
                'n_neighbors': [3, 5, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
        
            # Grilles pour la Classification (Binaire / Multi-classe) 
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest Classifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_leaf': [1, 2, 4]
            },
            'K-Neighbors Classifier': {
                'n_neighbors': [3, 5, 9],
                'weights': ['uniform', 'distance']
            },
        
            # Grilles pour la Classification Multi-label
            # N'oublie pas le préfixe 'estimator__' car le modèle est dans un MultiOutputClassifier
            'Random Forest Multi-label': {
                'estimator__n_estimators': [50, 100],
                'estimator__max_depth': [None, 10]
            },
            'K-Neighbors Multi-label': {
                'estimator__n_neighbors': [3, 5, 9],
                'estimator__weights': ['uniform', 'distance']
            }
        }
        self.trained_models = {}
        self.scores = {}
        self.problem_type = None
        self.X_test = None
        self.y_test = None
        self.best_model = None

    @staticmethod
    def load_dataset(folder_path: str):
        basename = os.path.basename(folder_path)
        data = pd.read_csv(f"{folder_path}/{basename}.data", sep="\s+", header=None, na_values='NaN', engine='python')
        data.columns = [f'feature_{i}' for i in range(data.shape[1])]
        
        solution = np.loadtxt(f"{folder_path}/{basename}.solution")
        
        with open(f"{folder_path}/{basename}.type", "r", encoding="utf-8") as f:
            ntypes = [line.strip() for line in f.readlines()]
        types = np.array(ntypes)      
        return data, solution, types


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

    def fit(self, folder:str):

        # --- 1) CHARGEMENT DES DONNEES ET IDENTIFICATION DU PROBLEME -----
        # Charger les données
        data, solution, types = self.load_dataset(folder)
        # Trouver le type de probleme
        self.problem_type = self.detect_task_type(solution)

        if self.problem_type is None:
            raise ValueError(f"Le problème n'a pas été identifié !")
        print(f"Probleme de type : {self.problem_type}")

        # --- 2) SEPARATION DES DONNEES (!! AVANT NETOYAGE DES DONNEES !!)-----
        # On separe les données en 80% entrainement et 20 restant pour les tests
        # random_state=42 permet de garantir que la séparation est toujours la même
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            data, solution, test_size=0.2, random_state=42
        )
        # -- 3) PRETRAITEMENT DES DONN2ES --
        # Recuperer les colonnes correspondent au differents type 
        cat_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Categorical"]
        num_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Numerical"]
        bin_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Binary"]
        # print(f"num_cols :{cat_cols}")
        if num_cols:
            # Imputation (Remplacement des données manquantes)
            num_imputer = SimpleImputer(strategy="median")
            # Calcule est remplace les valeurs manquante par la mediane
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            # remplace les valeurs manquante par la mediane sans re calculer !
            self.X_test[num_cols] = num_imputer.transform(self.X_test[num_cols])

             # Normalisation
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            self.X_test[num_cols] = scaler.transform(self.X_test[num_cols])
            
        if bin_cols:
            # imputation
            bin_imputer = SimpleImputer(strategy="most_frequent")
            X_train[bin_cols] = bin_imputer.fit_transform(X_train[bin_cols])
            self.X_test[bin_cols] = bin_imputer.transform(self.X_test[bin_cols])

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


        # --- 4) ENTRAÎNEMENT DES MODÈLES ---
        models_to_test = self.models[self.problem_type]
        print("\n--- Début de l'entraînement des modèles ---")
    
        for model_name, model in models_to_test:
            try:
                print(f"Optimisation du modèle : {model_name}")
                if model_name in self.param_grids:

                    if self.problem_type == 'regression':
                        scoring_metric = 'neg_mean_squared_error'
                    elif self.problem_type == 'multilabel_classification':
                        scoring_metric = 'f1_samples'
                    else:
                        scoring_metric = 'accuracy'
                        
                    # Créer l'objet GridSearchCV
                    # cv=3 signifie 3-fold cross-validation
                    # n_jobs=-1 utilise tous les cœurs CPU disponibles sur le cluster Skinner
                    grid_search = GridSearchCV(
                        model, 
                        self.param_grids[model_name], 
                        cv=3, 
                        scoring=scoring_metric, 
                        n_jobs=-1
                    )
                    
                    # Lancer la recherche
                    grid_search.fit(X_train, y_train)
                    print(f"   => Meilleurs paramètres trouvés : {grid_search.best_params_}")
                    # Le meilleur modèle est maintenant l'attribut best_estimator_
                    self.trained_models[model_name] = grid_search.best_estimator_
                else:
                    # Si pas de grille définie, on fait un entraînement simple
                    model.fit(X_train, y_train)
                    self.trained_models[model_name] = model
                    
                
                # # On entraîne le modèle uniquement sur les données d'entraînement
                # model.fit(X_train, y_train)
                # self.trained_models[model_name] = model
                
            except Exception as e:
                print(f"Erreur avec le modèle {model_name} : {e}")
        
        print("--- Entraînement terminé ! ---")


    def eval(self):
        if not self.trained_models:
            print(f"Aucun model n'a ete entrainé")
            return None
        
        for model_name, model in self.trained_models.items():
            # Prédiction sur les données de test
            y_pred = model.predict(self.X_test)
            
            score = 0
            metric = ""
            # Choisir une meilleure métrique pour chaque type de problème
            if self.problem_type == "regression":
                score = mean_squared_error(self.y_test, y_pred)
                metric = "MSE"
            elif self.problem_type == "multilabel_classification":
                # Ajoutez zero_division=0.0 pour dire à la fonction de mettre 0 sans afficher d'avertissement.
                score = f1_score(self.y_test, y_pred, average='samples', zero_division=0.0)
                metric = "F1-Score (samples)"
            else: # Binaire ou Multi-classe
                score = accuracy_score(self.y_test, y_pred)
                metric = "Accuracy"
            
            self.scores[model_name] = score
            print(f"Model: {model_name} et Score ({metric}): {score:.4f}")

        # Trouver le meilleur model
        if self.scores:
            if self.problem_type == "regression":
                # Pour la régression, on cherche le score le plus bas 
                best_model_name = min(self.scores, key=self.scores.get)
            else:
                # Pour la classification, on cherche le score le plus haut
                best_model_name = max(self.scores, key=self.scores.get)
            
            self.best_model = self.trained_models[best_model_name]
            print(f"\n------------------------------------------------\n")
            print(f"Meilleur modèle : {best_model_name} (Score: {self.scores[best_model_name]:.4f})")



        