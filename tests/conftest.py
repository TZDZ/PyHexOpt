from pathlib import Path

import meshio
import pytest


@pytest.fixture
def clean_square_mesh() -> meshio.Mesh:
    return meshio.read(Path(r"examples/Square_mesh/square.msh"))


@pytest.fixture
def clean_rot_square_mesh() -> meshio.Mesh:
    return meshio.read(Path(r"examples/Square_mesh/square_rot.msh"))


@pytest.fixture
def square_bad1_mesh() -> meshio.Mesh:
    return meshio.read(Path(r"examples/Square_mesh/square_bad_1.msh"))


@pytest.fixture
def square_rot_bad1_mesh() -> meshio.Mesh:
    return meshio.read(Path(r"examples/Square_mesh/square_rot_bad_1.msh"))


@pytest.fixture
def out_path() -> Path:
    out = Path(r"tests/out")
    if out.exists() is False:
        out.mkdir(parents=True)
    return out
