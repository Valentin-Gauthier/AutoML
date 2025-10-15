# 🤖 AutoML Simplifié

Ce projet implémente une classe Python simple nommée `AutoML`, conçue pour automatiser les étapes de base du Machine Learning : **chargement des données**, **prétraitement**, **sélection du type de problème**, **entraînement de plusieurs modèles** et **évaluation de leurs performances**.

---

## 🚀 Fonctionnalités Clés

La classe `AutoML` gère un flux de travail de Machine Learning de bout en bout avec les étapes suivantes :

### 1. Gestion des Modèles
La classe initialise un dictionnaire contenant une sélection de modèles populaires de la librairie `scikit-learn` pour quatre types de tâches :
* **Régression**
* **Classification Binaire**
* **Classification Multi-classe**
* **Classification Multi-étiquettes (Multi-label)**

### 2. Chargement des Données (`load_dataset`)
Une méthode statique pour charger les données à partir d'un chemin de dossier spécifique. Elle attend la présence de trois fichiers standardisés :
* `basename.data` : Contient les features (caractéristiques).
* `basename.solution` : Contient les cibles/labels (variables à prédire).
* `basename.type` : Définit le type de chaque colonne (`Categorical`, `Numerical`, `Binary`).

### 3. Détection du Type de Problème (`detect_task_type`)
Cette méthode statique analyse la structure des données cibles (`solution`) pour déterminer automatiquement le type de problème de Machine Learning à résoudre :
* **Régression** (valeurs continues)
* **Classification Binaire** (deux classes)
* **Classification Multi-classe** (plus de deux classes, une seule étiquette par instance)
* **Classification Multi-étiquettes** (plus de deux classes, plusieurs étiquettes possibles par instance)

### 4. Entraînement et Prétraitement (`fit`)
La méthode `fit` orchestre les étapes de préparation et d'entraînement :
1.  **Séparation des Données :** Division en ensembles d'entraînement (80%) et de test (20%) via `train_test_split`.
2.  **Prétraitement :**
    * **Imputation :** Remplacement des valeurs manquantes (`NaN`) en utilisant la **médiane** pour les colonnes numériques et la **valeur la plus fréquente** pour les colonnes binaires et catégorielles.
    * **Normalisation :** Mise à l'échelle des colonnes numériques via `StandardScaler`.
    * **Encodage :** Conversion des variables catégorielles en format numérique via `OneHotEncoder`.
3.  **Entraînement :** Entraînement de tous les modèles pertinents pour le type de problème détecté.

### 5. Évaluation et Sélection du Meilleur Modèle (`eval`)
La méthode `eval` évalue les performances de tous les modèles entraînés sur l'ensemble de test (`X_test` et `y_test`) :
* **Métriques utilisées :**
    * **Régression :** Erreur Quadratique Moyenne (**MSE** - *Mean Squared Error*).
    * **Classification Binaire/Multi-classe :** **Précision** (*Accuracy Score*).
    * **Classification Multi-étiquettes :** **Score F1 (samples)**.
* **Sélection du Meilleur Modèle :** Le modèle avec le meilleur score (le plus faible MSE pour la régression, le plus élevé pour la classification) est automatiquement sélectionné et stocké dans `self.best_model`.

---

## 🛠️ Dépendances

Ce code nécessite les bibliothèques Python suivantes :

```bash
pandas
numpy
scikit-learn