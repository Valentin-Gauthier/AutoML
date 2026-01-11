# AutoML

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)

Ce projet implémente une classe `AutoML` conçue pour automatiser l'intégralité du pipeline de Machine Learning.

###  Fonctionnalités Clés
*  **Chargement intelligent** des données.
*  **Prétraitement** et nettoyage automatique.
*  **Détection du type de problème** (Régression, Classification, Multi-label...).
*  **Optimisation des hyperparamètres** (via Nevergrad).
*  **Évaluation complète** des performances.


##  Structure des Données (Important)

Pour que l'AutoML fonctionne, vos fichiers doivent partager le même **nom de base** et se trouver dans le même dossier.
Vous devez fournir le **chemin complet vers le fichier `.data`**. L'outil déduira automatiquement les chemins vers la solution et les types.

Si votre dataset s'appelle `data_A`, vous devez avoir :

```text
/chemin/vers/dossier/
├── dataset_A.data        <-- Chemin à fournir à la méthode fit()
├── dataset_A.solution    # Les labels (y) - Déduit automatiquement
└── dataset_A.type        # Description des colonnes - Déduit automatiquement
```

##  Guide d'Utilisation

Voici un exemple complet pour lancer un entraînement et générer des prédictions :

```python
from AutoML.src.nevergrad.automl import AutoML

automl = AutoML()

data_dest_traindev="/info/corpus/ChallengeMachineLearning/data_test/data.data"
automl.fit(data_dest_traindev)
automl.eval()  # Renvoie des résultats d'évaluation

path_to_testset = "/info/corpus/ChallengeMachineLearning/data_test/data_test.data"
automl.predict(path_to_testset) #retourne une liste avec les predictions par donnée du dataset (donc par ligne de donnée)

```
