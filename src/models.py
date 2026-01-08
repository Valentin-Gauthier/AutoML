from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

MODELS_CONFIG = {
    'regression': [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge()),
        ('K-Neighbors Regressor', KNeighborsRegressor()),
        ('SVR', SVR()), # lent sur de gros datasets
        ('Random Forest Regressor', RandomForestRegressor(random_state=42)),
        ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=42)),
        ('MLP Regressor', TransformedTargetRegressor(regressor=MLPRegressor(random_state=42), transformer=StandardScaler())), # On normalise la cible
        ('Hist Gradient Boosting', HistGradientBoostingRegressor(random_state=42)),
        ('ElasticNet', ElasticNet()), # Ajout : Robuste si bcp de features corrélées
        ('Linear SVR', LinearSVR(random_state=42, dual='auto')), # Ajout : Ultra rapide pour gros datasets
        ('Extra Trees Regressor', ExtraTreesRegressor(random_state=42)), # Ajout : Variante rapide de RF
    ],
    'binary_classification': [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Hist Gradient Boosting', HistGradientBoostingClassifier(random_state=42)),
        ('K-Neighbors Classifier', KNeighborsClassifier()),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('SVC', SVC(random_state=42, probability=True)), 
        ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
        ('Bernoulli Naive Bayes', BernoulliNB()),
        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42)),
        ('MLP Classifier', MLPClassifier(random_state=42)),
        ('Linear SVC', LinearSVC(random_state=42, dual='auto')), # Ajout : Le roi des données Sparse/Texte
        ('Extra Trees Classifier', ExtraTreesClassifier(random_state=42)),
    ],
    'multiclass_classification': [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Hist Gradient Boosting', HistGradientBoostingClassifier(random_state=42)),
        ('K-Neighbors Classifier', KNeighborsClassifier()),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('Bernoulli Naive Bayes', BernoulliNB()),
        ('SVC', SVC(random_state=42, probability=True)), # Trop lent est très souvent pas le meilleur
        ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42)), # Hist Gradient Boosting -> bcp plus rapide et aussi/plus performant
        ('Linear SVC', LinearSVC(random_state=42, dual='auto')), # Bcp plus rapide que SVC
        ('Extra Trees Classifier', ExtraTreesClassifier(random_state=42)),
    ],
    'multilabel_classification': [
        ('Random Forest Classifier Multi-label', MultiOutputClassifier(RandomForestClassifier(random_state=42))), # tres long
        ('K-Neighbors Classifier Multi-label', MultiOutputClassifier(KNeighborsClassifier())),
        ('Gradient Boosting Classifier Multi-label', MultiOutputClassifier(GradientBoostingClassifier(random_state=42))),
        ('Hist Gradient Boosting Multi-label', MultiOutputClassifier(HistGradientBoostingClassifier(random_state=42))),
        ('MLP Multi-label', MLPClassifier(random_state=42)),
        ('Ridge Classifier', MultiOutputClassifier(RidgeClassifier())),
        ('Linear SVC Multi-label', MultiOutputClassifier(LinearSVC(random_state=42, dual='auto'))),
        ('Extra Trees Classifier Multi-label', MultiOutputClassifier(ExtraTreesClassifier(random_state=42))), # Plus rapide que RF
    ]
}

# ==============================================================================
# RÉSUMÉ DES FORCES (P) ET FAIBLESSES (N) DES MODÈLES
# ==============================================================================
#
# 1. Modèles Linéaires (Linear Regression, Ridge, Logistic Regression)
#    [P] Ultra rapides, interprétables, excellents "baselines" (points de comparaison).
#    [N] Incapables de capturer des relations non-linéaires complexes.
#
# 2. K-Neighbors (KNN)
#    [P] Capte des formes très complexes sans entraînement préalable ("Lazy learning").
#    [N] Devient très lent et inefficace quand le nombre de colonnes (dimensions) augmente.
#
# 3. Support Vector Machines (SVR, SVC)
#    [P] Très précis en haute dimension, efficace sur des petits jeux de données complexes.
#    [N] Complexité cubique : devient extrêmement lent (voire inutilisable) au-delà de 10 000 lignes.
#
# 4. Random Forest
#    [P] Très robuste, gère bien le sur-apprentissage, ne demande pas de mise à l'échelle (scaling).
#    [N] Modèle lourd en mémoire (RAM) et prédiction lente (doit interroger des centaines d'arbres).
#
# 5. Gradient Boosting (Standard)
#    [P] Souvent le plus précis sur les données tabulaires (gagne souvent les compétitions).
#    [N] Entraînement séquentiel (lent) et difficile à paralléliser.
#
# 6. Hist Gradient Boosting
#    [P] Version optimisée moderne : 100x plus rapide que le GB standard, gère les valeurs manquantes.
#    [N] Moins interprétable ("Boîte noire") et demande parfois beaucoup de données pour briller.
#
# 7. Naive Bayes (GaussianNB, BernoulliNB)
#    [P] Entraînement éclair, gère parfaitement les matrices creuses (Sparse) et le texte.
#    [N] Fait une hypothèse "naïve" d'indépendance des variables, souvent moins précis que les arbres.
#
# 8. Réseaux de Neurones (MLP)
#    [P] Capable de modéliser n'importe quelle fonction complexe et gère le Multi-label nativement (très rapide).
#    [N] "Boîte noire" difficile à régler (beaucoup d'hyperparamètres) et nécessite absolument des données normalisées.
#
# 9. Wrapper Multi-label (MultiOutputClassifier)
#    [P] Permet de transformer n'importe quel classifieur binaire en classifieur multi-label.
#    [N] Approche "force brute" : entraîne 1 modèle par label. Si vous avez 50 labels, c'est 50x plus lent.
# ==============================================================================