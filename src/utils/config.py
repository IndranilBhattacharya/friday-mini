import yaml
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    """
    Loads YAML configuration.
    
    In:
        path: str - path to YAML file (shape: none/scalar)
        
    Out:
        Dict[str, Any] - parsed configuration dictionary (shape: none/scalar)
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
