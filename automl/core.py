# Fichier: automl/core.py
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV

# Imports relatifs internes
from .data import load_dataset
from .utils import detect_task_type, get_scoring_metric, compute_metrics
from .models import MODELS_CONFIG

class AutoML:

    def __init__(self):
        # models
        self.models = MODELS_CONFIG
        
        # Config des hyperparametres (Chemin vers src/ à la racine du projet)
        try:
            with open("src/models_config.yaml", 'r') as f:
                self.param_grids = yaml.safe_load(f)
        except FileNotFoundError:
            print("Attention: 'src/models_config.yaml' introuvable. Optimisation désactivée.")
            self.param_grids = {}
        
        self.trained_models = {}
        self.scores = {}
        self.task_type = None
        self.X_test = None
        self.y_test = None
        self.best_model = None
        self.best_params = {}

    def fit(self, folder:str, test_size:float=0.2, verbose:bool=False):
        """
        Charger les données, trouver le type de problème, preparer les données et tester les modèles.
        """
        # 1) Charger le dataset
        data, solution, types = load_dataset(folder)
        
        # 2) identifier le type de problème
        self.task_type = detect_task_type(solution)
        if verbose:
            print(f"-> Tache de type {self.task_type} detecté.")

        # 3) Preparation des données
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            data, solution, test_size=test_size, random_state=42
        )
        
        cat_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Categorical"]
        num_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Numerical"]
        bin_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Binary"]

        # Traitement Numérique
        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            self.X_test[num_cols] = num_imputer.transform(self.X_test[num_cols])
            
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            self.X_test[num_cols] = scaler.transform(self.X_test[num_cols])

        # Traitement Binaire
        if bin_cols:
            bin_imputer = SimpleImputer(strategy="most_frequent")
            X_train[bin_cols] = bin_imputer.fit_transform(X_train[bin_cols])
            self.X_test[bin_cols] = bin_imputer.transform(self.X_test[bin_cols])

        # Traitement Catégoriel
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            self.X_test[cat_cols] = cat_imputer.transform(self.X_test[cat_cols])
        
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_train_encoded = ohe.fit_transform(X_train[cat_cols])
            X_test_encoded = ohe.transform(self.X_test[cat_cols])

            encoded_cols = ohe.get_feature_names_out(cat_cols)
            X_train_encoded_df = pd.DataFrame(X_train_encoded, index=X_train.index, columns=encoded_cols)
            X_test_encoded_df = pd.DataFrame(X_test_encoded, index=self.X_test.index, columns=encoded_cols)
        
            X_train = X_train.drop(cat_cols, axis=1)
            self.X_test = self.X_test.drop(cat_cols, axis=1)
        
            X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
            self.X_test = pd.concat([self.X_test, X_test_encoded_df], axis=1)

        # 5) Entrainement des models
        if self.task_type in self.models:
            models_to_test = self.models[self.task_type]
        else:
            print(f"Type de tâche {self.task_type} non supporté dans MODELS_CONFIG")
            models_to_test = []

        if verbose:
            print("\n-> Modèles sélectionnés :")
            for name, _ in models_to_test:
                print(f"   - {name}")

        scoring_metric = get_scoring_metric(self.task_type)

        for model_name, model in models_to_test:
            try:
                if verbose:
                    print(f"\n... Entraînement de : {model_name}")
                
                if model_name in self.param_grids:
                    # Recherche d'hyperparamètres
                    grid_search = GridSearchCV(
                        model, 
                        self.param_grids[model_name], 
                        cv=3, 
                        scoring=scoring_metric, 
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    
                    if verbose:
                        print(f"   => Meilleurs params : {grid_search.best_params_}")
                    
                    self.trained_models[model_name] = grid_search.best_estimator_
                    self.best_params[model_name] = grid_search.best_params_
                else:
                    # Entraînement simple sans grille
                    model.fit(X_train, y_train)
                    self.trained_models[model_name] = model
                    self.best_params[model_name] = "Défaut"
                
            except Exception as e:
                print(f"Erreur avec {model_name} : {e}")

    def eval(self):
        if not self.trained_models:
            print(f"Aucun modèle entraîné.")
            return None

        print("\n================================================")
        print(" RÉSULTATS DÉTAILLÉS")
        print("================================================")
        
        main_scores = {} 

        for model_name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            
            # Calcul complet des métriques
            metrics_dict = compute_metrics(self.y_test, y_pred, self.task_type)
            self.scores[model_name] = metrics_dict
            
            print(f"\n[Modèle : {model_name}]")
            for metric_name, val in metrics_dict.items():
                print(f"  {metric_name:<10} : {val:.4f}")
            
            # Choix du score pour déterminer le vainqueur
            if self.task_type == "regression":
                main_scores[model_name] = metrics_dict.get("R2", -float('inf'))
            else:
                # On privilégie le F1-Score au Accuracy
                main_scores[model_name] = metrics_dict.get("F1-Score", 0)

        # Sélection du meilleur modèle
        if main_scores:
            best_model_name = max(main_scores, key=main_scores.get)
            
            self.best_model = self.trained_models[best_model_name]
            best_metrics = self.scores[best_model_name]
            best_params = self.best_params[best_model_name]
            
            print(f"\n================================================")
            print(f" STRONGEST : {best_model_name}")
            print(f"================================================")
            for k, v in best_metrics.items():
                print(f"{k:<15}: {v:.4f}")
            print(f"------------------------------------------------")
            print(f"Hyperparamètres : {best_params}")
            print(f"================================================")