from src.nevergrad.automl import AutoML

root_path = "/info/corpus/ChallengeMachineLearning"
datasets_letters = "ABCDEFGHIJK"

for letter in datasets_letters:
    
    dataset_name = f"data_{letter}"  

    current_data_path = os.path.join(root_path, dataset_name, f"{dataset_name}.data")
    
    print(f"############################################### Traitement du dataset : {dataset_name} #############################################")
    print(f"-> Chargement de : {current_data_path}")

    automl = AutoML(
        budget=40,                        
        timeout_min=10,                   
        num_workers=10,
        mem_gb=12,                      
        verbose=True
    )
    
    try:
        automl.fit(current_data_path)
        automl.eval()
        
    except Exception as e:
        print(f"ERREUR lors du traitement de {dataset_name}: {e}")
        continue