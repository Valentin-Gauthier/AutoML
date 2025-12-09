# Fichier: automl/__init__.py
from .core import AutoML

# Création d'une instance unique (Singleton) pour conserver l'état
_instance = AutoML()

def fit(folder, test_size=0.2, verbose=False):
    """
    Interface simplifiée demandée par le sujet.
    Appelle la méthode fit() de l'instance unique.
    """
    return _instance.fit(folder, test_size, verbose)

def eval():
    """
    Interface simplifiée demandée par le sujet.
    Appelle la méthode eval() de l'instance unique.
    """
    return _instance.eval()