import os
from typing import List


def get_spack_extension_paths() -> List[str]:
    return [os.path.join(os.path.dirname(__file__), "spack-solver-benchmark")]


def get_spack_config_dir() -> str:
    return []
