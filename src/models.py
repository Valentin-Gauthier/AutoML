from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB

MODELS_CONFIG = {
    'regression': [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge()),
        ('K-Neighbors Regressor', KNeighborsRegressor()),
        ('SVR', SVR()),
        ('Random Forest Regressor', RandomForestRegressor(random_state=42)),
        ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=42))
    ],
    'binary_classification': [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
        ('K-Neighbors Classifier', KNeighborsClassifier()),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('SVC', SVC(random_state=42, probability=True)),
        ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42))
    ],
    'multiclass_classification': [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')),
        ('K-Neighbors Classifier', KNeighborsClassifier()),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('SVC', SVC(random_state=42, probability=True)),
        ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42))
    ],
    'multilabel_classification': [
        ('RF Multi-label', MultiOutputClassifier(RandomForestClassifier(random_state=42))),
        ('KN Multi-label', MultiOutputClassifier(KNeighborsClassifier())),
        ('GB Multi-label', MultiOutputClassifier(GradientBoostingClassifier(random_state=42)))
    ]
}