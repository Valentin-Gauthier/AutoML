import numpy as np
# Assure-toi que ton code est accessible, par exemple en installant ton paquet en mode éditable
from src.automl import AutoML 

# On crée une instance de la classe pour pouvoir appeler la méthode statique
automl_instance = AutoML()

def test_detect_task_type():
    """
    Teste unitairement la fonction detect_task_type avec des données simulées.
    """
    # --- Cas 1 : Régression ---
    # Données simulées : un vecteur 1D avec des valeurs continues/multiples
    solution_regression = np.array([150.5, 99.0, 243.1, 500.0])
    assert automl_instance.detect_task_type(solution_regression) == "regression"
    print("Test Régression OK")

    # --- Cas 2 : Classification Binaire ---
    # Données simulées : un vecteur 1D avec seulement 0 et 1
    solution_binary = np.array([0, 1, 1, 0, 1])
    assert automl_instance.detect_task_type(solution_binary) == "binary_classification"
    print("Test Binaire OK")

    # --- Cas 3 : Classification Multi-classe (one-hot) ---
    # Données simulées : une matrice 2D avec un seul '1' par ligne
    solution_multiclass = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    assert automl_instance.detect_task_type(solution_multiclass) == "multiclass_classification"
    print("Test Multi-classe OK")

    # --- Cas 4 : Classification Multi-label ---
    # Données simulées : une matrice 2D avec potentiellement plusieurs '1' par ligne
    solution_multilabel = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    assert automl_instance.detect_task_type(solution_multilabel) == "multilabel_classification"
    print("Test Multi-label OK")