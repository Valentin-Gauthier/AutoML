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

Pour que l'AutoML fonctionne, vos fichiers doivent suivre une convention de nommage stricte.
L'outil détecte automatiquement les extensions. **Ne fournissez que le chemin racine.**

Si votre dataset s'appelle `data_A`, vous devez avoir :

```text
/chemin/vers/dossier/data_A
├── data_A.data       # Les features (X)
├── data_A.solution   # Les labels (y) - Requis pour l'entraînement
└── data_A.types      # Description des colonnes (Optionnel)
```

##  Guide d'Utilisation

Voici un exemple complet pour lancer un entraînement et générer des prédictions :

```python
from AutoML.src.nevergrad.automl import AutoML

# Dossier de train
path_to_train = "/info/corpus/ChallengeMachineLearning/data_test/data_A"
# Dossier de test (L'outil cherchera data_test.data uniquement)
path_to_test = "/info/corpus/ChallengeMachineLearning/data_test/data_test"

automl = AutoML()

# Lancement du pipeline complet
automl.fit(path_to_train)


# Affichage des scores des meilleurs modèles
automl.eval()

# Génération des prédictions finales
predictions = automl.predict(path_to_test)

```
