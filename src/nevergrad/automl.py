import numpy as np
import pandas as pd
import yaml 
import os
import submitit 
import shutil
import nevergrad as ng
import tempfile
import time
from scipy.sparse import coo_matrix, issparse, csr_matrix
from typing import Tuple, Any, Union, Callable, List, Dict

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, make_scorer, classification_report, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.base import clone


from .trainer import ModelTrainer
from src.models import MODELS_CONFIG

# Ignorer les warnings
import warnings  
warnings.filterwarnings("ignore", message="Ignoring since timer was stopped before starting")


class AutoML:
    """Defines parameters for AutoML using submitit & nevergrad.

    Example:
        >>> from AutoML.src.nevergrad.automl import AutoML
        >>> automl = AutoML()

    Args:
        test_size (float):
            Fraction of the dataset to include in the test split. 
            Must be between 0.0 and 1.0. Default to 0.2.
        random_state (int):
            Seed used to ensure reproducible results. Default to 42.
        scoring (str):
            Metric to optimize (e.g. 'f1', 'r2'); 'auto' infers it from the task. 
            Default to auto.
        n_folds (int):
            Number of folds used for cross-validation. Default to 3.
        feature_selection_threshold (int or float):
            Threshold of number of features above which Feature Selection (SelectKBest) 
            is triggered to reduce dimensionality. Defaults to infinity (no selection by default).
        models_config_path (str):
            Path of the models configuration YAML file.
        budget (int):
            Total budget for the nevergrad optimization (total iterations).
            Default to 40.
        num_workers (int):
            Number of parallel workers (jobs) launching with submitit. Default to 10.
        timeout_min (int):
            Maximum duration in minutes allowed for each job (before killed). Default to 15.
        mem_gb (int):
            RAM requested per job in Gigabytes. Default to 8.
        cpus_per_task (int):
            CPU cores requested per job. Default to 1.
        submitit_log_folder (str):
            Path where submitit logs will be saved.
        cluster_partition (str):
            Cluster name to execute submitit. Default to gpu (Skinner).
        verbose (bool):
            If True, enables detailed logging output. Default to False.
    """

    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 scoring: str = "auto",
                 n_folds: int = 3,
                 feature_selection_threshold: Union[int, float] = float('inf'),
                 models_config_path: str = "src/nevergrad/models_config.yaml",
                 budget: int = 40,
                 num_workers: int = 10,
                 timeout_min: int = 15,
                 mem_gb: int = 8,
                 cpus_per_task: int = 1,
                 submitit_log_folder: str = "automl_logs",
                 cluster_partition : str = "gpu",
                 verbose: bool = False,
                 ):

        # data sciences params
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = scoring
        self.n_folds = n_folds
        self.feature_selection_threshold = feature_selection_threshold
        self.models_config_path = models_config_path

        # optimisation params
        self.budget = budget

        # cluster/submitit params
        self.num_workers = num_workers
        self.timeout_min = timeout_min
        self.mem_gb = mem_gb
        self.cpus_per_task = cpus_per_task
        self.submitit_log_folder = submitit_log_folder
        self.cluster_partition = cluster_partition

        # utils
        self.verbose = verbose

        # models params
        self.models = MODELS_CONFIG
        self.trained_models = {}
        self.scores = {}
        self.task_type = None
        self.X_test = None
        self.y_test = None
        self.best_model = None
        self.best_params = {}

        if os.path.exists(self.models_config_path):
            with open(self.models_config_path, 'r') as f:
                self.models_params = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"[init] Config file not found at: {self.models_config_path}")

        if self.submitit_log_folder is None:
            self.submitit_log_folder = "automl_logs"

        if os.path.exists(self.submitit_log_folder):
            try:
                shutil.rmtree(self.submitit_log_folder)
            except Exception as e:
                print(f"[init] Warning: the provided folder cant be clean : {e}")

        try:
            os.makedirs(self.submitit_log_folder, exist_ok=True)
            log_path = os.path.abspath(self.submitit_log_folder)
            
        except Exception as e:
            raise ValueError(f"[init] Invalid log folder path '{self.submitit_log_folder}': {e}")

        self.executor = submitit.AutoExecutor(folder=log_path)

        self.executor.update_parameters(
            timeout_min=self.timeout_min,     
            mem_gb=self.mem_gb,             
            cpus_per_task=self.cpus_per_task,     
            slurm_partition=cluster_partition
        )

    @staticmethod
    def load_sparse_matrix(data_path:str) -> csr_matrix:
        """Loads a sparse dataset from a text file. 
        Element are space-separated string in the format 'column_index:value'.

        Exemple format:
            0:1.5 12:0.4 
            2:0.5 45:1.1
        
        Args:
            data_path (str): Path the the data file.
        
        Returns:
            csr_matrix
        """
        rows = []
        cols = []
        data = []
        try:
            with open(data_path, 'r') as f:
                for row_idx, line in enumerate(f):
                    elements = line.strip().split()

                    for item in elements:
                        try:
                            # 'column_index:value'
                            idx_str, val_str = item.split(":")
                            rows.append(row_idx)
                            cols.append(int(idx_str))
                            data.append(float(val_str))
                        except ValueError:
                            print(f"Warning: Malformed item '{item} at line {row_idx}. Skipped.")
                            continue

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {data_path}")
        # convert in CSR (better for calculs)
        return coo_matrix((data, (rows, cols))).tocsr()
    
    def load_dataset(self, data_path:str) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Load the dataset files (.data, .solution, .type) based on the .data file path.
    
        Args:
            data_path (str): Full path to the .data file.
                             Ex: '/info/corpus/ChallengeMachineLearning/data_test/data.data'
                             The function assumes that .solution and .type files 
                             share the same base name and directory.
        
        Returns:
            Tuple: A tuple containing:
                - X (pd.DataFrame | csr_matrix): the features.
                - y (np.array): the solution array.
                - types (np.array | None): Column types if dense, else None.
        """
        data_path = os.path.normpath(data_path)

        base_path, _ = os.path.splitext(data_path)
    
        sol_path = f"{base_path}.solution"
        types_path = f"{base_path}.type" 
    
        basename = os.path.basename(base_path)

        data = None
        solution = None
        types = None

        try:
            # X
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"[load dataset] Data file not found: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()

            if ":" in first_line:
                if self.verbose:
                    print(f"[load dataset] Loading Sparse dataset: {basename}")
                data = self.load_sparse_matrix(data_path)
            else:
                if self.verbose:
                    print(f"[load dataset] Loading Dense dataset: {basename}")
                data = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN", engine="python")
                data.columns = [f"feature_{i}" for i in range(data.shape[1])]
                
                # types
                if os.path.exists(types_path):
                    with open(types_path, 'r', encoding='utf-8') as f:
                        types = np.array([line.strip() for line in f.readlines()])
                # else:
                #     raise FileNotFoundError(f"[load dataset] Type file not found: {types_path}")
            # y      
            if os.path.exists(sol_path):
                solution = np.loadtxt(sol_path)
            # else:
            #     raise FileNotFoundError(f"[load dataset] Solution file not found: {sol_path}")
            
            return data, solution, types
        
        except Exception as e:
            raise RuntimeError(f"[load dataset] Error loading dataset from {data_path}") from e
            
    def get_scoring_metric(self, task_type:str) -> Union[str, Callable]:
        """Determines the best Scikit-Learn scoring metric based on the task type.
        If a custom metric was provided in __init__, it overrides the automatic selection.

        Args:
            task_type (str): The detected task type (ex: 'regression')

        Returns:
            str : A valid Scikit-Learn scoring string (ex: 'f1') or a scorer object.
        """
        if self.scoring != "auto":
            if self.verbose:
                print(f"[scoring metric] Using user-defined metric: {self.scoring}")
                return self.scoring
            
        if task_type == "binary_classification":
            # ROC_AUC est plus robuste que l'accuracy car indépendant du seuil de décision
            return "roc_auc"
        elif task_type == "multiclass_classification":
            # F1_Macro est crucial si les classes sont déséquilibrées (évite de favoriser la classe majoritaire)
            return "f1_macro"
        
        elif task_type == "multilabel_classification":
            # Average='samples' calcule le F1 pour chaque ligne (instance) puis fait la moyenne.
            # C'est la métrique standard pour le multi-label.
            return make_scorer(f1_score, average='samples', zero_division=0)
        
        elif task_type == "regression":
            # R2 est standardisé (1.0 = parfait, 0.0 = modèle naïf), plus lisible que le MSE
            return "r2"
        
        else:
            # Fallback de sécurité
            if self.verbose:
                print(f"[get_scoring] Warning: Unknown task type '{task_type}'. Defaulting to 'accuracy'.")
            return "accuracy"
        
    @staticmethod
    def detect_task_type(solution:np.ndarray) -> str:
        """Infer the machine learning task type based on the target array (y) shape and values.

        Heuristics used:
            - 2D array + One-Hot (rows sum to 1) -> 'multiclass_classification'
            - 2D array + Multi-Hot -> 'multilabel_classification'
            - 1D array + 2 unique values -> 'binary_classification'
            - 1D array + Integers -> 'multiclass_classification' (Label Encoding). < 20 labels
            - 1D array + Floats -> 'regression'

        Args:
            solution (np.ndarray): The target data (y).

        Returns:
            str: One of 'binary_classification', 'multiclass_classification', 
                 'multilabel_classification', or 'regression'.
        """
        # 2D (matrix)
        if solution.ndim > 1:
            row_sums = np.sum(solution, axis=1)
            is_one_hot = np.allclose(row_sums, 1) and np.all(np.isin(solution, [0, 1]))

            if is_one_hot:
                return "multiclass_classification"
            else:
                return "multilabel_classification"

        # 1D
        else:
            unique_values = np.unique(solution)
            num_unique = len(unique_values)

            if num_unique == 2:
                return "binary_classification"
            
            is_integer = np.all(np.mod(solution, 1) == 0)

            # limite de 20 arbitraire pour eviter les cas comme le dataset Iris (y : 0, 1, 2)
            if is_integer and num_unique < 20: 
                return "multiclass_classification"
            else:
                return "regression"
    
    def dataset_analysis(self, solution:np.ndarray) -> None:
        """Prints statistical analysis of the target variable (distribution, balance, etc.).
        
        Args:
            solution (np.ndarray): The target array (y).
        """
        if not self.verbose:
            return None
        
        n_samples = len(solution)
        print(f"\n    Dataset Target Analysis ({n_samples} samples)")

        if self.task_type == "regression":
            print(f" Type: Regression")
            print(f"  - Min:    {np.min(solution):.4f}")
            print(f"  - Max:    {np.max(solution):.4f}")
            print(f"  - Mean:   {np.mean(solution):.4f}")
            print(f"  - Median: {np.median(solution):.4f}")
            print(f"  - StdDev: {np.std(solution):.4f}")

        elif self.task_type == "multilabel_classification":
            counts = np.sum(solution, axis=0)
            n_labels = len(counts)

            print(f"   Type: Multi-label ({n_labels} labels)")
            print(f" - Average labels per sample: {np.mean(np.sum(solution, axis=1)):.2f}")

            sorted_indices = np.argsort(-counts)
            for i in sorted_indices:
                count = counts[i]
                ratio = count / n_samples
                print(f" -> Label {i:<3}: {int(count):<6} ({ratio:.2%})")

        elif "classification" in self.task_type:
            if solution.ndim > 1:
                y_flat = np.argmax(solution, axis=1)
            else:
                y_flat = solution.astype(int)

            unique, counts = np.unique(y_flat, return_counts=True)
            is_imbalanced = (np.max(counts) / np.min(counts)) > 2
            balance_msg = "IMBALANCED" if is_imbalanced else "Balanced"
            print(f" Type: {self.task_type.replace('_', ' ').title()} [{balance_msg}]")

            for u, c in zip(unique, counts):
                print(f"  -> Class {u:<3}: {c:<6} samples ({c/n_samples:.2%})")
        else:
            print(f"[dataset analysis] Warning: Unknown task type '{self.task_type}'. Cannot analyze distribution.")
        print("\n")
            
    def _filter_models(self, X, y=None) -> List[Tuple[str, Any]]:
        """Dynamically selects models compatible with the dataset characteristics
        (size, sparsity, dimensions, task constraints).

        Heuristics applied:
            1. Sparse Data: Exclude models that require dense arrays (e.g. GaussianNB).
            2. Large Dataset (>10k rows): Exclude cubic complexity models (SVC, SVR).
            3. High Dimension (>500 cols): Exclude distance-based models (KNN).
            4. Massive Multi-label (>5 labels): Exclude 'One-vs-Rest' wrappers, keep native multi-output (MLP, Ridge).

        Args:
            X (pd.DataFrame | csr_matrix): Feature matrix.
            y (np.array, optional): Target vector/matrix.
            verbose (bool): Whether to print exclusion logs.

        Returns:
            List[Tuple[str, Any]]: A list of (model_name, model_instance) tuples to train.
        """
        all_models = self.models.get(self.task_type, [])
        models_to_keep = []
        
        n_rows, n_cols = X.shape
        is_data_sparse = issparse(X)

        # Vérification rapide si données négatives (pour Naive Bayes)
        has_negative_values = False
        if not is_data_sparse and hasattr(X, 'min'):
             if X.min().min() < 0:
                 has_negative_values = True
        
        n_labels = 1
        n_classes = 2
        
        if y is not None:
            if self.task_type == "multilabel_classification" and y.ndim > 1:
                n_labels = y.shape[1]
                
            elif self.task_type == "multiclass_classification":
                try:
                    n_classes = len(np.unique(y))
                except:
                    n_classes = 5
                

        if self.verbose:
            print(f"\n [filter models] Dynamic Model Filtering ---")
            print(f" Dataset Info: {n_rows} rows, {n_cols} cols, Sparse={is_data_sparse}")
            if n_labels > 1:
                print(f" Target Info : {n_labels} labels (Multi-label)")
            if n_classes > 2:
                print(f" Target Info : {n_classes} classes (Multiclass)")

        for name, model in all_models:
            reason = None

            # Compatibilité Sparse
            if is_data_sparse and (name in ["Gaussian Naive Bayes"] or "Gradient Boosting" in name):
                reason = "Incompatible with Sparse data"

            # Compatibilité Données Négatives
            elif has_negative_values and "Naive Bayes" in name:
                 reason = "Incompatible with negative values (StandardScaler)"
                
            # Complexité Cubique (Seulement SVC/SVR classiques, pas les Linear)
            elif name in ["SVC", "SVR"]:
                if n_rows > 5000:
                    reason = f"Too slow for {n_rows} rows (Limit: 5000)"
                elif n_rows > 2000 and n_cols > 500:
                    reason = f"Too slow for matrix {n_rows}x{n_cols}"

            # Haute Dimension (Distance based)
            elif n_cols > 500 and "Neighbors" in name:
                reason = f"Ineffective in high dimensions ({n_cols} cols)"

            # Multi-label Massif (>20 labels)
            elif self.task_type == "multilabel_classification" and n_labels > 5:
                
                fast_multioutput = ["MLP", "Ridge", "Linear SVC"]
                is_fast = any(fast in name for fast in fast_multioutput)
                
                if not is_fast:
                      reason = f"Too heavy for {n_labels} labels (requires {n_labels} models)"
                    
            elif self.task_type == "multiclass_classification" and n_classes >= 5:
                if "Gradient Boosting" in name:
                     reason = f"Too slow for {n_classes} classes (requires {n_classes} trees per iteration)"
                
            if reason:
                if self.verbose:
                    print(f" [EXCLUDED] {name:.<35} : {reason}")
            else:
                models_to_keep.append((name, model))
        
        if self.verbose:
            print(f" -> Models selected: {len(models_to_keep)} / {len(all_models)}\n")
            
        return models_to_keep

    def _config_to_nevergrad(self, grid_config: Dict[str, Any]) -> ng.p.Instrumentation:
        """Converts a dictionary configuration (from YAML) into a Nevergrad search space.

        Mapping Logic:
            - List [...]  -> ng.p.Choice([...]) : The optimizer chooses one value from the list.
            - Value (int/str/float) -> Constant : The value is fixed for all iterations.

        Example:
            Input:  {'n_estimators': [10, 100], 'criterion': 'gini'}
            Output: Instrumentation(n_estimators=Choice([10, 100]), criterion='gini')

        Args:
            grid_config (Dict[str, Any]): Dictionary containing hyperparameter grids.

        Returns:
            ng.p.Instrumentation: The parameterization object ready for the optimizer.
        """
        ng_dict = {}
        
        if not grid_config:
            return ng.p.Instrumentation()

        for param, value in grid_config.items():
            
            if isinstance(value, list):
                ng_dict[param] = ng.p.Choice(value)
            
            elif isinstance(value, dict):
                dist_type = value.get("type", "float")
                lower = value.get("min")
                upper = value.get("max")

                if dist_type == "int":
                    ng_dict[param] = ng.p.Scalar(lower=lower, upper=upper).set_integer_casting()
                
                elif dist_type == "log":
                    ng_dict[param] = ng.p.Log(lower=lower, upper=upper)

                else: 
                    ng_dict[param] = ng.p.Scalar(lower=lower, upper=upper)
            else:
                ng_dict[param] = value
        
        return ng.p.Instrumentation(**ng_dict)

    def fit(self, dataset_folder:str) -> None:
        """Orchestrates the complete AutoML training pipeline.

        This method executes the following steps sequentially:
        1.  **Data Loading**: Ingests data (Sparse/Dense) and detects the machine learning task type.
        2.  **Preprocessing**: Applies specific transformations:
            - Sparse: MaxAbsScaler.
            - Dense: Imputation (Median/Freq), Scaling (Standard), and One-Hot Encoding.
        3.  **Feature Selection**: Reduces dimensionality using statistical tests
            if the number of features exceeds `self.feature_selection_threshold`.
        4.  **Dynamic Filtering**: Excludes models incompatible with the dataset (size, sparsity, etc.).
        5.  **Hyperparameter Optimization**: Launches distributed search using Nevergrad & Submitit.
        6.  **Final Training**: Retrains the best configuration found on the full training set.

        Args:
            dataset_folder (str): Path to the directory containing the dataset files.
                                  The directory must contain files with the same basename:
                                  - `*.data` (Features)
                                  - `*.solution` (Targets)
                                  - `*.type` (Column types, required for dense data)
        """
        data, solution, types = self.load_dataset(dataset_folder)
        self.task_type = self.detect_task_type(solution)
        if self.verbose:
            print(f"[fit] {self.task_type} task detected.")
            
        self.dataset_analysis(solution)
        
        # convert y: 2D to 1D
        if self.task_type == "multiclass_classification" and solution.ndim > 1:
            solution = np.argmax(solution, axis=1)

        # data preparation
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            data, solution, test_size=self.test_size, random_state=self.random_state
        )

        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.bin_imputer = None
        self.ohe = None
        self.selector = None

        self.cat_cols = None
        self.num_cols = None
        self.bin_cols = None


        if issparse(X_train):
            self.scaler = MaxAbsScaler()
            X_train = self.scaler.fit_transform(X_train)
            self.X_test = self.scaler.transform(self.X_test)
        else:
            self.cat_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Categorical"]
            self.num_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Numerical"]
            self.bin_cols = [f"feature_{i}" for i, t in enumerate(types) if t == "Binary"]

            if self.num_cols:
                self.num_imputer = SimpleImputer(strategy="median")
                X_train[self.num_cols] = self.num_imputer.fit_transform(X_train[self.num_cols])
                self.X_test[self.num_cols] = self.num_imputer.transform(self.X_test[self.num_cols])

                self.scaler = StandardScaler()
                X_train[self.num_cols] = self.scaler.fit_transform(X_train[self.num_cols])
                self.X_test[self.num_cols] = self.scaler.transform(self.X_test[self.num_cols])

            if self.bin_cols:
                self.bin_imputer = SimpleImputer(strategy="most_frequent")
                X_train[self.bin_cols] = self.bin_imputer.fit_transform(X_train[self.bin_cols])
                self.X_test[self.bin_cols] = self.bin_imputer.transform(self.X_test[self.bin_cols])

            if self.cat_cols:
                self.cat_imputer = SimpleImputer(strategy="most_frequent")
                X_train[self.cat_cols] = self.cat_imputer.fit_transform(X_train[self.cat_cols])
                self.X_test[self.cat_cols] = self.cat_imputer.transform(self.X_test[self.cat_cols])
            
                # Encodage One-Hot
                self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                X_train_encoded = self.ohe.fit_transform(X_train[self.cat_cols])
                X_test_encoded = self.ohe.transform(self.X_test[self.cat_cols])

                encoded_cols = self.ohe.get_feature_names_out(self.cat_cols)
                X_train_encoded_df = pd.DataFrame(X_train_encoded, index=X_train.index, columns=encoded_cols)
                X_test_encoded_df = pd.DataFrame(X_test_encoded, index=self.X_test.index, columns=encoded_cols)
            
                X_train = X_train.drop(self.cat_cols, axis=1)
                self.X_test = self.X_test.drop(self.cat_cols, axis=1)
            
                X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
                self.X_test = pd.concat([self.X_test, X_test_encoded_df], axis=1)

        # data analysis
        n_rows, n_cols = X_train.shape

        if n_cols > self.feature_selection_threshold:
            if self.verbose:
                print(f"[fit] Features threshold exceeded ({n_cols} > {self.feature_selection_threshold}).")
                print(f"[fit] Reducing to the top {self.feature_selection_threshold} features...")
                print(f"[fit] Warning: Only kept {self.feature_selection_threshold / n_cols:.2%} of the features !")
    
            if self.task_type == "regression":
                score_func = f_regression
            else:
                if issparse(X_train):
                    score_func = chi2
                else:
                    score_func = f_classif
            
            selector = SelectKBest(score_func=score_func, k=self.feature_selection_threshold)
            try:
                X_train_reduced = selector.fit_transform(X_train, y_train)
                self.X_test = selector.transform(self.X_test)
    
                X_train = X_train_reduced
                if self.verbose:
                    print(f"[fit] Reduction done. New shape: {X_train.shape}")
            except Exception as e:
                print(f"[fit] Warning: Feature selection failed: {e}. Keeping original features.")

        # training models
        candidates = self.models.get(self.task_type, [])
        if self.verbose:
            print(f"\n[fit] Candidate models loaded for task '{self.task_type}':")
            for i, (name, _) in enumerate(candidates, start=1):
                print(f"   {i}. {name}")

        # filtering models
        models_to_test = self._filter_models(X_train, y_train)
        
        for model_name, model_instance in models_to_test:

            if model_name not in self.models_params:
                print(f"[LOCAL] No config found. Training with default params: {model_name}")
                model_instance.fit(X_train, y_train)
                self.trained_models[model_name] = model_instance
                continue
                
            model_config = self.models_params[model_name]

            try:
                if self.verbose:
                    print(f"[CLUSTER] Optimizing: {model_name}...")
                    
                trainer = ModelTrainer(
                    model_class=model_instance,
                    X=X_train,
                    y=y_train,
                    scoring=self.get_scoring_metric(self.task_type),
                    cv=self.n_folds
                )
                params_space = self._config_to_nevergrad(model_config)
                optimizer = ng.optimizers.TwoPointsDE(parametrization=params_space, budget=self.budget, num_workers=self.num_workers)
                start_opt = time.time()
                recommendation = optimizer.minimize(trainer, executor=self.executor, batch_mode=True)
                end_opt = time.time()
                best_params = recommendation.value[1]
                if self.verbose:
                    duration_opt = end_opt - start_opt
                    print(f"[CLUSTER] Success ({duration_opt:.1f}s). Best params: {best_params}")
                    print(f"[fit] Retraining final model on full data...")
                    
                final_model = clone(model_instance)
                final_model.set_params(**best_params)

                # modele direct (ex: Random Forest)
                if hasattr(final_model, 'n_jobs'):
                    final_model.set_params(n_jobs=-1)
                    
                #  Wrapper (ex: OneVsRest ou MultiOutputClassifier)
                if hasattr(final_model, 'estimator'):
                    if hasattr(final_model.estimator, 'n_jobs'):
                        final_model.estimator.set_params(n_jobs=-1)
                        
                start_fit = time.time()
                final_model.fit(X_train, y_train)
                end_fit = time.time()

                if self.verbose:
                    print(f"[fit] Final training done in {end_fit - start_fit:.1f}s.")

                self.trained_models[model_name] = final_model
                self.best_params[model_name] = best_params

            except Exception as e:
                print(f"[CLUSTER] Error optimizing {model_name}: {e}")
                print(f"[LOCAL] Fallback: Training locally with default params.")
                try:
                    model_instance.fit(X_train, y_train)
                    self.trained_models[model_name] = model_instance
                except Exception as ex_local:
                    print(f"[LOCAL] Failed to train {model_name}: {ex_local}. Model Skipped.")
                    continue

    def eval(self) -> None:
        """
        Evaluates all trained models on the test set and selects the best one.

        Metrics used for selection:
            - Regression: R2 Score (higher is better).
            - Classification (Binary/Multiclass): F1-Score Macro.
            - Multilabel: F1-Score Samples.

        Updates:
            self.best_model: The model instance with the highest score.
            self.best_params: The hyperparameters of the best model.
        """
        if not self.trained_models:
            print("[eval] No models trained yet.")
            return

        if self.X_test is None or self.y_test is None:
             print("[eval] Test data missing. Cannot evaluate.")
             return

        print(f"\n[eval] --- Detailed Results on Test Set ({len(self.y_test)} samples) ---")
        
        final_results = [] 

        for model_name, model in self.trained_models.items():
            print(f"\n > Model: {model_name}")
            
            try:
                y_pred = model.predict(self.X_test)
                
                # regression
                if self.task_type == "regression":
                    mse = mean_squared_error(self.y_test, y_pred)
                    r2 = r2_score(self.y_test, y_pred)
                    
                    print(f"   - MSE: {mse:.4f}")
                    print(f"   - R2 : {r2:.4f}")
                    
                    final_results.append((model_name, r2))

                # multi-label
                elif self.task_type == "multilabel_classification":
                    f1_samples = f1_score(self.y_test, y_pred, average='samples', zero_division=0.0)
                    f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0.0)
                    
                    print(f"   - F1 (Samples): {f1_samples:.4f} (Quality per instance)")
                    print(f"   - F1 (Macro)  : {f1_macro:.4f} (Quality per label)")

                    if self.verbose:
                        print("\n[Classification Report]")
                        print(classification_report(self.y_test, y_pred, zero_division=0.0))
                    
                    final_results.append((model_name, f1_samples))

                # binary / multi-class
                else:
                    acc = accuracy_score(self.y_test, y_pred)
                    f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0.0)
                    
                    print(f"   - Accuracy : {acc:.4f}")
                    print(f"   - F1 Macro : {f1_macro:.4f} (Balanced metric)")
                    
                    if self.verbose:
                        print("\n[Classification Report]")
                        print(classification_report(self.y_test, y_pred, zero_division=0.0))
                    
                    final_results.append((model_name, f1_macro))
            
            except Exception as e:
                print(f"[eval] Prediction failed for {model_name}: {e}")

        if not final_results:
            print("[eval] No valid results obtained.")
            return

        best_model_name, best_score = max(final_results, key=lambda x: x[1])
        self.best_model = self.trained_models[best_model_name]
        
        best_params = self.best_params.get(best_model_name, "Default/Local Params")

        print(f"\n==================================================")
        print(f" BEST MODEL : {best_model_name}")
        print(f" Score      : {best_score:.4f}")
        print(f" Params     : {best_params}")
        print(f"==================================================\n")              






    def predict(self, dataset_folder:str) -> np.ndarray:
        """
        """
        if self.best_model is None:
            raise ValueError("[predict] You must call fit() before predict().")

        if self.verbose:
            print(f"[predict] Loading data from {dataset_folder}...")

        X_test, _, _ = self.load_dataset(dataset_folder)

        if issparse(X_test):
            if self.scaler:
                X_test = self.scaler.transform(X_test)
        else:
            if not isinstance(X_test, pd.DataFrame):
               X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])

            if self.num_cols and self.num_imputer:
                X_test[self.num_cols] = self.num_imputer.transform(X_test[self.num_cols])
                X_test[self.num_cols] = self.scaler.transform(X_test[self.num_cols])
            
            if self.bin_cols and self.bin_imputer:
                X_test[self.bin_cols] = self.bin_imputer.transform(X_test[self.bin_cols])
                
            if self.cat_cols and self.cat_imputer:
                X_test[self.cat_cols] = self.cat_imputer.transform(X_test[self.cat_cols])
                
                if self.ohe:
                    X_encoded = self.ohe.transform(X_test[self.cat_cols])
                    encoded_cols = self.ohe.get_feature_names_out(self.cat_cols)
                    X_encoded_df = pd.DataFrame(X_encoded, index=X_test.index, columns=encoded_cols)
                    
                    X_test = X_test.drop(self.cat_cols, axis=1)
                    X_test = pd.concat([X_test, X_encoded_df], axis=1)
            

        if self.selector:
            print(f"[predict] Applying feature selection...")
            X_test = self.selector.transform(X_test)

        
            
        if self.verbose:
            model_name = type(self.best_model).__name__
            if hasattr(self.best_model, "estimator"):
                inner_name = type(self.best_model.estimator).__name__
                model_name = f"{inner_name} (wrapped in {model_name})"
            print(f"[predict] Predicting using {model_name}...")
            
        predictions = self.best_model.predict(X_test)
        
        return predictions








