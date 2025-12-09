import automl

# Remplacez ceci par le chemin réel vers vos données sur le cluster Skinner
dataset_path = "/info/corpus/ChallengeMachineLearning/data_A" 

print("--- Démarrage de l'AutoML ---")

# Lancement de l'entraînement
automl.fit(dataset_path, verbose=True)

# Lancement de l'évaluation
automl.eval()