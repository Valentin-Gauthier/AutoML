from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
MODELS_CONFIG = {
    'regression': [
        ('Linear Regression', LinearRegression(n_jobs=-1)), 
        ('Ridge', Ridge()), 
        ('K-Neighbors Regressor', KNeighborsRegressor(n_jobs=-1)), 
        ('SVR', SVR()), 
        ('Random Forest Regressor', RandomForestRegressor(random_state=42, n_jobs=-1)), 
        ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=42)) 
    ],
    'binary_classification': [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)), 
        ('K-Neighbors Classifier', KNeighborsClassifier(n_jobs=-1)), 
        ('Gaussian Naive Bayes', GaussianNB()), 
        ('SVC', SVC(random_state=42, probability=True)), 
        ('Random Forest Classifier', RandomForestClassifier(random_state=42, n_jobs=-1)), 
        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42)) 
    ],
    'multiclass_classification': [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', n_jobs=-1)), 
        ('K-Neighbors Classifier', KNeighborsClassifier(n_jobs=-1)),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('SVC', SVC(random_state=42, probability=True)),
        ('Random Forest Classifier', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42))
    ],
    'multilabel_classification': [
        ('RF Multi-label', MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=-1), n_jobs=None)),

        ('KN Multi-label', MultiOutputClassifier(KNeighborsClassifier(n_jobs=-1), n_jobs=None)),

        ('GB Multi-label', MultiOutputClassifier(GradientBoostingClassifier(random_state=42), n_jobs=-1))
    ]
}