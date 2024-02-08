import os
from pathlib import Path


def dir_to_project_root():
    current_dir = Path(os.getcwd())
    project_dir = [p for p in current_dir.parents if p.parts[-1] == "rotten_tomatoes"][
        0
    ]
    return project_dir
