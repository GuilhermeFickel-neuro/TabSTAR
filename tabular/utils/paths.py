import os
from os.path import join
import re

CACHE_DIR = ".tabular_cache"



_PRETRAIN_DIR = join(CACHE_DIR, "pretrain")
_FINETUNE_DIR = join(CACHE_DIR, "finetune")
_BASELINES_DIR = join(CACHE_DIR, "baselines")
_DATASET_DIR = join(CACHE_DIR, "datasets")


def pretrain_exp_dir(exp_name: str) -> str:
    return join(_PRETRAIN_DIR, exp_name)

def get_model_path(run_name: str, is_pretrain: bool) -> str:
    if is_pretrain:
        main_dir = pretrain_exp_dir(run_name)
    else:
        main_dir = downstream_run_dir(run_name, is_tabstar=True)
    return main_dir


def pretrain_args_path(exp_name: str) -> str:
    return join(pretrain_exp_dir(exp_name), "pretrain_args.json")


def downstream_run_dir(run_name: str, is_tabstar: bool) -> str:
    if is_tabstar:
        main_dir = _FINETUNE_DIR
    else:
        main_dir = _BASELINES_DIR
    return join(main_dir, run_name)


def train_results_path(run_name: str, is_tabstar: bool) -> str:
    return join(downstream_run_dir(run_name, is_tabstar=is_tabstar), "results.json")


def dataset_run_properties_dir(run_num: int, train_examples: int) -> str:
    return join(_DATASET_DIR, f"run{run_num}_n{train_examples}")

def properties_path(data_dir: str) -> str:
    return join(data_dir, "properties.json")

def create_dir(path: str, is_file: bool = False):
    if is_file:
        path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)

def sanitize_filename_component(name: str) -> str:
    """Sanitizes a string to be a safe component in a filename or directory name."""
    if not isinstance(name, str):
        name = str(name)
    
    # Replace specific problematic characters (like *, 🌟) with underscores
    name = name.replace('*', '_')
    # Using a more generic name for emoji replacement, can be expanded
    name = name.replace('🌟', '_star_') 
    name = name.replace('✨', '_star_') # Explicit unicode for glowing star
    name = name.replace('🌟', '_star_') # Another star emoji variant

    # General sanitization for other common problematic characters in filenames
    # Windows reserved: < > : " / \ | ? *
    # We've handled * and emoji. '/' is handled by os.path.join.
    # Let's replace <, >, ", |, ?, \ with underscore.
    # ':' can be tricky; on Windows, it's a drive separator. For now, let's replace it too if it's not part of a drive letter.
    # Simpler: replace known bad characters.
    name = re.sub(r'[<>:"\|?]', '_', name)
    
    # Replace multiple consecutive underscores with a single underscore
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores that might result from replacements
    name = name.strip('_')
    # Ensure name is not empty after sanitization
    if not name:
        name = "sanitized_name" # Default if everything gets stripped
    return name
