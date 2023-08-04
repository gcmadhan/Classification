import os
import yaml
from pathlib import Path
from Classification.logging import logger
from box import ConfigBox
from box.exceptions import BoxValueError

def read_yaml(path_to_yaml: Path)-> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

