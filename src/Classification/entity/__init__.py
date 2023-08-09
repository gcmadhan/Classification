from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    data_file_path: Path
    target_feature: str

@dataclass(frozen=True)
class Outputfile:
    path: Path
    
@dataclass(frozen=True)
class PreProcessing:
    cat_trans: list[str]
    num_trans: list[str]

@dataclass(frozen=True)
class Models:
    models: list[str]
    

