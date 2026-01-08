# src/trainer.py
import submitit
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

class ModelTrainer(submitit.helpers.Checkpointable):
    """

    """
    def __init__(self, model_class, X, y, scoring, cv=3):
        # Note : Sur de très gros datasets, on passerait le chemin du fichier 
        self.model_class = model_class
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        
    def __call__(self, **parameters):
        try:
            model = clone(self.model_class)
            model.set_params(**parameters)
        except Exception as e:
            print(f"Erreur init modèle avec {parameters} : {e}")
            return self._return_bad_score()
        
        try:
            scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=1)
            mean_score = np.mean(scores)
        except Exception as e:
            print(f"Erreur entrainement : {e}")
            # retourne la pire note possible 
            return float('inf')

        return self._process_score(mean_score)

   
    def _process_score(self, score):
        """Inverse le score si nécessaire (car Nevergrad minimise toujours)"""
      
        if isinstance(self.scoring, str):
             # Liste des métriques où "plus c'est haut, mieux c'est"
             maximize_metrics = ["accuracy", "f1", "r2", "roc_auc", "precision", "recall"]
             if any(m in self.scoring for m in maximize_metrics):
                 return -score
             return score
        # Si c'est un objet Scorer, c'est généralement une métrique à maximiser
        return -score