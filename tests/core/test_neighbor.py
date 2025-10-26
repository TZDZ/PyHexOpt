# tests/core/test_neighbor.py
from pathlib import Path

import jax.numpy as jnp
import meshio
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.neighbor import build_element_adjacency, build_node_to_elements_sorted, get_element_neighborhood


@pytest.fixture
def simple_mesh_cells():
    """
    Minimal mesh where:

    elem0 ─ elem1 ─ elem2
        \
         elem3

    - elem0 and elem1 share a face (4 nodes in common).
    - elem1 and elem2 share a (different) face (4 nodes in common) that does NOT belong to elem0.
    - elem3 shares a small connection (node(s)) with elem0 only.
    """
    cells = jnp.array(
        [
            # elem0: base element
            [0, 1, 2, 3, 4, 5, 6, 7],  # elem0
            # elem1: shares face with elem0 via nodes {1,2,5,6}, has extra nodes {8,9,10,11}
            [1, 8, 9, 2, 5, 10, 11, 6],  # elem1
            # elem2: shares face with elem1 via nodes {8,9,10,11} only (so not adjacent to elem0)
            [8, 18, 19, 9, 10, 20, 21, 11],  # elem2
            # elem3: connects to elem0 through node 0 (and 4) — a small connection
            [0, 16, 17, 18, 4, 19, 20, 21],  # elem3
        ],
        dtype=jnp.int32,
    )
    return cells


def test_build_element_adjacency(simple_mesh_cells):
    adj = build_element_adjacency(simple_mesh_cells)

    # adjacency is square and boolean
    E = simple_mesh_cells.shape[0]
    assert adj.shape == (E, E)
    assert adj.dtype == jnp.bool_

    # symmetric and diagonal is False
    assert jnp.all(adj == adj.T)
    assert not jnp.any(jnp.diag(adj))

    # known neighbor pairs
    assert adj[0, 1] and adj[1, 0]  # share face
    assert adj[1, 2] and adj[2, 1]  # share face
    assert adj[0, 3] and adj[3, 0]  # share a node(s)
    assert not adj[2, 0]  # no direct connection between elem2 and elem0


def test_first_neighborhood(simple_mesh_cells):
    node_elem_index_data = build_node_to_elements_sorted(simple_mesh_cells)
    nodes, elems = get_element_neighborhood(simple_mesh_cells, node_elem_index_data, elem_index=0, depth=1)

    # should include elem0 and its direct neighbors (elem1 and elem3)
    assert set(elems.tolist()) == {0, 1, 3}

    # node list should contain all nodes used by these elements
    used_nodes = jnp.unique(simple_mesh_cells[elems].ravel())
    assert jnp.array_equal(jnp.sort(nodes), used_nodes)


def test_second_neighborhood(simple_mesh_cells):
    node_elem_index_data = build_node_to_elements_sorted(simple_mesh_cells)
    nodes, elems = get_element_neighborhood(simple_mesh_cells, node_elem_index_data, elem_index=0, depth=2)

    # second neighborhood should include 0,1,2,3
    assert set(elems.tolist()) == {0, 1, 2, 3}

    # includes all nodes used by those elements
    used_nodes = jnp.unique(simple_mesh_cells[elems].ravel())
    assert jnp.array_equal(jnp.sort(nodes), used_nodes)


def test_isolation_case():
    # Two isolated elements (no shared nodes)
    cells = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=jnp.int32,
    )

    adj = build_element_adjacency(cells)
    assert not jnp.any(adj)
    node_elem_index_data = build_node_to_elements_sorted(cells)

    nodes, elems = get_element_neighborhood(cells, node_elem_index_data, elem_index=0, depth=3)
    # Only the element itself should appear
    assert jnp.array_equal(elems, jnp.array([0]))
    used_nodes = jnp.unique(cells[0].ravel())
    assert jnp.array_equal(jnp.sort(nodes), used_nodes)


def test_8x8x8():
    msh = meshio.read(Path(r"examples/Square_mesh/8x8x8cube.msh"))
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    node_elem_index_data = build_node_to_elements_sorted(cells)

    nodes, elems = get_element_neighborhood(cells, node_elem_index_data, elem_index=456 - 1, depth=1)
    assert len(elems) == 8
    nodes, elems = get_element_neighborhood(cells, node_elem_index_data, elem_index=8 - 1, depth=1)
    assert len(elems) == 8
    nodes, elems = get_element_neighborhood(cells, node_elem_index_data, elem_index=7 - 1, depth=1)
    assert len(elems) == 12
    nodes, elems = get_element_neighborhood(cells, node_elem_index_data, elem_index=8 - 1, depth=2)
    assert len(elems) == 3**3
    nodes, elems = get_element_neighborhood(cells, node_elem_index_data, elem_index=8 - 1, depth=3)
    assert len(elems) == 4**3


if __name__ == "__main__":
    # test_8x8x8()
    pytest.main([__file__])
