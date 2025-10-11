import meshio
import numpy as np
import pytest

from pyhexopt.core.utils import get_boundary_nodes


def test_single_hex_boundary_nodes():
    # Define 8 cube vertices (unit cube)
    points = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ]
    )

    # One hexahedron cell using these 8 points
    cells = [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]

    mesh = meshio.Mesh(points=points, cells=cells)

    boundary_nodes = get_boundary_nodes(mesh)

    # For a single hex, all 8 nodes are boundary nodes
    expected = np.arange(8)

    assert np.array_equal(np.sort(boundary_nodes), expected)


def test_two_adjacent_hexes_shared_face():
    points = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
            [2, 0, 0],  # 8
            [2, 1, 0],  # 9
            [2, 0, 1],  # 10
            [2, 1, 1],  # 11
        ]
    )

    cells = [
        (
            "hexahedron",
            np.array(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7],  # left cube
                    [1, 8, 9, 2, 5, 10, 11, 6],  # right cube
                ]
            ),
        )
    ]

    mesh = meshio.Mesh(points=points, cells=cells)
    boundary_nodes = get_boundary_nodes(mesh)
    expected = np.arange(12)

    assert set(boundary_nodes) == set(expected)


def test_3x3():
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
    bnd_nodes = get_boundary_nodes(msh)
    assert len(bnd_nodes) == 56
    for n in (59, 60, 63, 64, 61, 62, 57, 58):
        assert n not in bnd_nodes


if __name__ == "__main__":
    pytest.main([__file__])
