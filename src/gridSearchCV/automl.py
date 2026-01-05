import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, issparse, csr_matrix
import os
import tqdm
import yaml
from src.models import MODELS_CONFIG

# Normaliser des données
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
# Remplacer les valeurs manquantes
from sklearn.impute import SimpleImputer
# One Hot Encoder poour les données 
from sklearn.preprocessing import OneHotEncoder
# division des données en données d'entrainement et de tests
from sklearn.model_selection import train_test_split
# Evaluer les models
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, make_scorer, classification_report
# Optimiser les hyperparametres
from sklearn.model_selection import GridSearchCV
# permet de garder les features qui permettent réellement de prédire Y
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


class AutoML:

    def __init__(self):
        # models
        self.models = MODELS_CONFIG
        # config des hyperparametres
        with open("src/gridSearchCV/models_config.yaml", 'r') as f:
            self.param_grids = yaml.safe_load(f)
        
        self.trained_models = {}
        self.scores = {}
        self.task_type = None
        self.X_test = None
        self.y_test = None
        self.best_model = None
        self.best_params = {}

    @staticmethod
    def load_sparse_matrix(data_path:str):
        """Charger une matrice creuse"""
        rows = []
        cols = []
        data = []

        with open(data_path, 'r') as f:
            for row_idx, line in enumerate(f):
                elements = line.strip().split()

                for item in elements: # 24:1
                    idx_str, val_str = item.split(":")
                    rows.append(row_idx)
                    cols.append(int(idx_str))
                    data.append(float(val_str))
        
        return coo_matrix((data, (rows, cols))).tocsr()
                    
        
    def load_dataset(self, folder_path:str):
        """
        Charger les trois fichiers du dataset comprenant :
            - data_X.data -> les données
            - data_X.solution -> les résultats
            - data_X.type -> les types des colonnes 

        Args:
            - folder_path (str): Chemin du dossier du dataset.
        """
        basename = os.path.basename(folder_path)
        data_path = f"{folder_path}/{basename}.data"

        try:
            # Lire la premiere ligne et lire le format
            with open(data_path, 'r') as f:
                first_line = f.readline()

                if ":" in first_line:
                    # Format Sparse
                    data = self.load_sparse_matrix(data_path)
                    types = None # dans les matrices sparces il y a que des valeurs numériques
                else:
                    # Format Dense
                    data = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN", engine="python")
                    data.columns = [f"feature_{i}" for i in range(data.shape[1])]
                     
                    with open(f"{folder_path}/{basename}.type", 'r', encoding="utf-8") as f:
                        types = np.array([line.strip() for line in f.readlines()])
        except Exception as e:
            print(f"Erreur de chargement des données : {e}")
            return None, None, None
        
        # Charge les solutions
        solution = np.loadtxt(f"{folder_path}/{basename}.solution")
        return data, solution, types

    
    def get_scoring_metric(self, task_type: str):
        if task_type == "binary_classification":
            return "roc_auc" # ou 'accuracy'
        elif task_type == "multiclass_classification":
            return "f1_macro" # Mieux que accuracy si classes déséquilibrées
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

    def dataset_analyses(self, solution:np.array):
        """
        Analyse le datasets
        """
        print(f"\n--- Analyse de la distribution du dataset complet ({len(solution)} lignes) ---")
        
        if self.task_type == "regression":
            print(f"Cible (y) : Min={np.min(solution):.2f}, Max={np.max(solution):.2f}, Moyenne={np.mean(solution):.2f}")
            print(f"Écart-type : {np.std(solution):.2f}")
            
        elif self.task_type == "multilabel_classification":
            # Pour le multi-label, on compte les '1' dans chaque colonne
            counts = np.sum(solution, axis=0)
            total = solution.shape[0]
            print(f"Nombre de labels : {len(counts)}")
            for i, c in enumerate(counts):
                # Affiche le nombre d'occurences et le pourcentage
                print(f" -> Label {i} : {int(c)} exemples ({c/total:.2%})")
                
        elif self.task_type == "multiclass_classification":
            # Si c'est du One-Hot (2D), on convertit en 1D pour compter
            if solution.ndim > 1:
                y_temp = np.argmax(solution, axis=1)
            else:
                y_temp = solution
            
            unique, counts = np.unique(y_temp, return_counts=True)
            for u, c in zip(unique, counts):
                print(f" -> Classe {u} : {c} exemples ({c/len(y_temp):.2%})")
                
        elif self.task_type == "binary_classification":
            unique, counts = np.unique(solution, return_counts=True)
            for u, c in zip(unique, counts):
                print(f" -> Classe {u} : {c} exemples ({c/len(solution):.2%})")


    def _filter_models(self, X, y=None, verbose=False):
        """
        Sélectionne dynamiquement les modèles compatibles avec le dataset
        en fonction de sa taille, sa sparsité et ses dimensions.
        """
        # On récupère la liste complète des candidats
        all_models = self.models[self.task_type]
        models_to_keep = []
        
        # Analyse des métadonnées du dataset
        n_rows, n_cols = X.shape
        is_data_sparse = issparse(X)
        
        # Pour le multi-label
        n_labels = 1
        if self.task_type == "multilabel_classification" and y is not None:
             if y.ndim > 1:
                n_labels = y.shape[1]

        if verbose:
            print(f"\n--- Filtrage dynamique des modèles ---")
            print(f"Dataset : {n_rows} lignes, {n_cols} colonnes, Sparse={is_data_sparse}")
            if n_labels > 1: print(f"Labels cibles : {n_labels}")

        for name, model in all_models:
            reason = None
            
            # Règle 1 : Incompatibilité SPARSE
            # (Note: HistGradientBoosting peut gérer le sparse mais crash souvent par manque de RAM si conversion forcée)
            if is_data_sparse and name in ["Gaussian Naive Bayes", "Hist Gradient Boosting", "Hist Gradient Boosting Multi-label"]:
                reason = "Incompatible avec format Sparse"

            # Règle 2 : Trop lent sur gros volume (> 10k lignes) -> SVC/SVR
            elif n_rows > 10000 and name in ["SVC", "SVR"]:
                reason = f"Trop lent pour {n_rows} lignes (Complexité Cubique)"

            # Règle 3 : Malédiction de la dimensionnalité (> 500 colonnes) -> KNN
            elif n_cols > 500 and "Neighbors" in name:
                reason = f"Inefficace en haute dimension ({n_cols} colonnes)"

            # Règle 4 : Explosion du Multi-label (> 20 labels) -> Wrappers
            elif self.task_type == "multilabel_classification" and n_labels > 20:
                # On ne garde que les modèles natifs (MLP, Ridge)
                if "MLP" not in name and "Ridge" not in name:
                     reason = f"Trop lourd pour {n_labels} labels (nécessite {n_labels} modèles)"

            # Décision
            if reason:
                if verbose:
                    print(f" [EXCLU] {name:.<35} : {reason}")
            else:
                models_to_keep.append((name, model))
        
        if verbose:
            print(f"Modèles retenus : {len(models_to_keep)} / {len(all_models)}")
            
        return models_to_keep
        

    def fit(self, folder:str, test_size:float=0.2, verbose:bool=False):
        """
        Charger les données, trouver le type de problème, preparer les données et tester les différents modèle de sklearn associé.
        
        Args:
            - folder (str): Chemin vers le dossier du dataset.
            - test_size (float): Taille du batch de test ]0, 1[
            - verbose (bool): Active certain log
        """
        # Charger le dataset
        data, solution, types = self.load_dataset(folder)

        # Identifier le type de probleme
        self.task_type = self.detect_task_type(solution)
        if verbose:
            print(f"-> Tache de type {self.task_type} detecté.")
            self.dataset_analyses(solution)
            

        if self.task_type == "multiclass_classification" and solution.ndim > 1:
            if verbose: 
                print("-> Conversion de la cible y : One-Hot (2D) vers Labels (1D)")
            # np.argmax renvoie l'index de la valeur max (le 1) pour chaque ligne
            solution = np.argmax(solution, axis=1)

        # --- Préparation des données ---
        # Split des données
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            data, solution, test_size=test_size, random_state=42
        )
        # Traitement des données
        if issparse(X_train):
            # on ne centre pas les données sinon cela tranformerait tous les 0 -> explosion taille mémoire
            scaler = MaxAbsScaler()

            X_train = scaler.fit_transform(X_train)
            self.X_test = scaler.transform(self.X_test)
            # Pas besoin d'imputer ni de gerer des données catégoriques etc

        else:
            cat_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Categorical"]
            num_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Numerical"]
            bin_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Binary"]

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
        n_rows, n_cols = X_train.shape
        
        # Si on a trop de colonnes denses (ex: > 1000), ça va ramer
        if n_cols > 1000 and not issparse(X_train): 
            if verbose:
                print(f"\n-> Trop de colonnes DENSES détectées ({n_cols})...")
                print("   Lancement de la sélection statistique des meilleures features...")
            
            # 1. Choisir le score selon le problème
            if self.task_type == "regression":
                score_func = f_regression
            else:
                score_func = f_classif # Fonctionne pour classification

            # 2. On garde les 500 colonnes les plus corrélées à la cible
            selector = SelectKBest(score_func=score_func, k=500)
            
            try:
                X_train_reduced = selector.fit_transform(X_train, y_train)
                self.X_test = selector.transform(self.X_test)
                
                # On remplace par la version réduite
                X_train = X_train_reduced
                
                if verbose:
                    print(f"   -> Réduction terminée : On garde les {X_train.shape[1]} colonnes les plus utiles.")
            except Exception as e:
                print(f"   -> Echec de la réduction de dimension : {e}")
        

        # 5) Entrainement des models correspondant
        if verbose:
            print("\n-> Choix des models correspondant : \n")
            models = self.models[self.task_type]
            models_name = [name for name, model in models]
            for i in range(len(models_name)):
                print(f"{i+1} - {models_name[i]}")
                
        models_to_test = self._filter_models(X_train, y_train, verbose=verbose)
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

                    jobs_count = -1
                    if model_name in ["SVC", "K-Neighbors Classifier Multi-label"]:
                        jobs_count = 1

                    # Créer l'objet GridSearchCV
                    # cv=3 signifie 3-fold cross-validation
                    # n_jobs=-1 utilise tous les cœurs CPU disponibles sur le cluster Skinner
                    grid_search = GridSearchCV(
                        model, 
                        self.param_grids[model_name], 
                        cv=3, 
                        scoring=self.get_scoring_metric(self.task_type), 
                        n_jobs=jobs_count,
                        verbose=3
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

        print("\n -> Résultats détaillés : \n")
        
        # On garde une liste pour comparer à la fin
        final_results = []

        for model_name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            
            print(f"--- Rapport pour : {model_name} ---")
            
            if self.task_type == "regression":
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = model.score(self.X_test, self.y_test) # Le R2 est souvent plus parlant
                print(f"MSE: {mse:.4f} | R2: {r2:.4f}")
                final_results.append((model_name, r2)) # On choisira le meilleur selon R2

            elif self.task_type == "multilabel_classification":
                f1_samples = f1_score(self.y_test, y_pred, average='samples', zero_division=0.0)
                f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0.0)
                
                print(f"F1-Score (Samples) : {f1_samples:.4f} (Qualité par ligne)")
                print(f"F1-Score (Macro)   : {f1_macro:.4f} (Qualité moyenne des labels)")
                print(classification_report(self.y_test, y_pred, zero_division=0.0))
                
                final_results.append((model_name, f1_samples))

            else: # Binary ou Multiclass
                # L'accuracy est souvent insuffisante
                acc = accuracy_score(self.y_test, y_pred)
                f1_weighted = f1_score(self.y_test, y_pred, average='weighted', zero_division=0.0)
                f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0.0)
                
                print(f"Accuracy: {acc:.4f}")
                print(f"F1-Score (Weighted): {f1_weighted:.4f} (Biaisé vers la majorité)")
                print(f"F1-Score (Macro)   : {f1_macro:.4f} (Équitable)")
                # Affiche la précision et le rappel pour chaque classe
                print(classification_report(self.y_test, y_pred, zero_division=0.0))
                
                final_results.append((model_name, f1_macro)) # On juge sur le F1, plus robuste

        # --- Sélection du meilleur modèle ---
        # On trie pour prendre le score le plus élevé (valable pour R2 et F1)
        best_model_name, best_score = max(final_results, key=lambda x: x[1])
        
        self.best_model = self.trained_models[best_model_name]
        self.best_params[best_model_name] = self.best_params.get(best_model_name)

        print(f"\n================================================")
        print(f" MEILLEUR MODÈLE : {best_model_name}")
        print(f" Score retenu    : {best_score:.4f}")
        print(f" Hyperparamètres : {self.best_params[best_model_name]}")
        print(f"================================================")



        