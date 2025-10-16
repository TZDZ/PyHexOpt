from pathlib import Path

import meshio
import pytest


@pytest.fixture
def clean_square_mesh() -> meshio.Mesh:
    return meshio.read(Path(r"examples/Square_mesh/square.msh"))


@pytest.fixture
def square_bad1_mesh_path() -> Path:
    return Path(r"examples/Square_mesh/square_bad_1.msh")


@pytest.fixture
def out_path() -> Path:
    return Path(r"tests/out")
