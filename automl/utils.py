# Fichier: automl/utils.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    make_scorer
)

def detect_task_type(solution: np.array) -> str:
    """Trouve le type de probleme (classification, regression etc)"""
    # Si il y a plus de 1 dimensions
    if solution.ndim > 1:
        # Compter le nombre de 1 par ligne
        # Si un seul 1 par ligne alors multiclasse sinon multi-labels
        if np.all(np.sum(solution, axis=1) == 1):
            return "multiclass_classification"
        else:
            return "multilabel_classification"
    else:
        sflat = solution.flatten()
        unique_values = np.unique(sflat)
        
        if len(unique_values) == 2:
            return "binary_classification"
        else:
            return "regression"

def get_scoring_metric(task_type: str):
    """Retourne la métrique utilisée pour l'OPTIMISATION (GridSearch)"""
    if task_type == "binary_classification":
        return "roc_auc"
    elif task_type == "multiclass_classification":
        return "f1_weighted"
    elif task_type == "multilabel_classification":
        return make_scorer(f1_score, average='samples', zero_division=0)
    elif task_type == "regression":
        return "r2"
    else:
        return "accuracy"

def compute_metrics(y_true, y_pred, task_type: str) -> dict:
    """
    Calcule un ensemble complet de métriques pour l'évaluation finale.
    """
    scores = {}

    # --- REGRESSION ---
    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        scores["MSE"] = mse
        scores["RMSE"] = np.sqrt(mse)
        scores["MAE"] = mean_absolute_error(y_true, y_pred)
        scores["R2"] = r2_score(y_true, y_pred)

    # --- CLASSIFICATION ---
    else:
        scores["Accuracy"] = accuracy_score(y_true, y_pred)
        
        # 'binary' pour binaire, 'weighted' pour multiclasse
        avg = 'binary' if task_type == 'binary_classification' else 'weighted'
        
        if task_type == 'multilabel_classification':
            avg = 'samples'
            
        scores["F1-Score"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        
        if task_type != 'multilabel_classification':
            scores["Precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            scores["Recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0)

    return scores