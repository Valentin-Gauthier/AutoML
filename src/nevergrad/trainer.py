# src/trainer.py
import submitit
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

class ModelTrainer(submitit.helpers.Checkpointable):
    """
    L'ouvrier qui s'exécute sur le cluster.
    Il reçoit un modèle et des hyperparamètres, il s'entraîne et renvoie le score.
    """
    def __init__(self, model_class, X, y, scoring, cv=3):
        # On stocke les données et la configuration
        # Note : Sur de très gros datasets, on passerait le chemin du fichier 
        # plutôt que les données brutes pour éviter de surcharger le réseau (pickle)
        self.model_class = model_class
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        
    def __call__(self, **parameters):
        try:
            # 1. On clone le modèle pour partir d'une base propre
            model = clone(self.model_template)
            
            # 2. On applique les paramètres via set_params
            # C'est LA méthode qui gère 'estimator__n_neighbors' correctement
            model.set_params(**parameters)
            
        except Exception as e:
            print(f"Erreur init modèle avec {parameters} : {e}")
            return self._return_bad_score()
        
        try:
            # 3. Entraînement
            scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=1)
            mean_score = np.mean(scores)
        except Exception as e:
            print(f"Erreur entrainement : {e}")
            return self._return_bad_score()

        # 4. Retour (Gestion Minimisation/Maximisation)
        return self._process_score(mean_score)

    def _return_bad_score(self):
        """Renvoie un score infini (mauvais) selon le type de métrique"""
        if isinstance(self.scoring, str):
            if "error" in self.scoring or "loss" in self.scoring:
                return float('inf') # Pour une erreur, l'infini est le pire
            return -float('inf')    # Pour un score, -l'infini est le pire
        
        # Si c'est un objet (make_scorer), on suppose qu'on veut maximiser (ex: F1)
        return -float('inf')

    def _process_score(self, score):
        """Inverse le score si nécessaire (car Nevergrad minimise toujours)"""
        # Si c'est un string
        if isinstance(self.scoring, str):
             # Liste des métriques où "plus c'est haut, mieux c'est"
             maximize_metrics = ["accuracy", "f1", "r2", "roc_auc", "precision", "recall"]
             if any(m in self.scoring for m in maximize_metrics):
                 return -score
             return score
        
        # Si c'est un objet Scorer, c'est généralement une métrique à maximiser
        return -score