from src.nevergrad.automl import AutoML

base_path = "/info/corpus/ChallengeMachineLearning/data_"
datasets_letters = "BCDEFGHIJK"

for letter in datasets_letters:
    
    current_data_dest = f"{base_path}{letter}"
    
    print(f"############################################### Traitement du dataset : data_{letter} #############################################")
    automl = AutoML(
        budget=40,
        feature_selection_threshold=800,
        timeout_min=10,
        num_workers=10,
        mem_gb=8,
        verbose=True
    )
    automl.fit(current_data_dest)
    automl.eval()