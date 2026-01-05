from src import automl
import importlib
importlib.reload(automl)
from src.automl import AutoML


base_path = "/info/corpus/ChallengeMachineLearning/data_"
datasets_letters = "ABCDEFGHIJK"

for letter in datasets_letters:
    current_data_dest = f"{base_path}{letter}"
    
    print(f"############################################### Traitement du dataset : data_{letter} #############################################")
    model = AutoML()
    model.fit(current_data_dest, verbose=True)
    model.eval()