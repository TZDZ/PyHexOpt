import jax
import jax.numpy as jnp


def build_element_adjacency(cells: jnp.ndarray) -> jnp.ndarray:
    """
    Build boolean adjacency matrix for hexahedral mesh.

    Two elements are adjacent iff they share at least one node.
    """
    E, Nn = cells.shape

    elem_ids = jnp.repeat(jnp.arange(E), Nn)
    node_ids = cells.flatten()
    order = jnp.argsort(node_ids)
    elem_ids = elem_ids[order]
    node_ids = node_ids[order]

    unique_nodes, node_starts, node_counts = jnp.unique(node_ids, return_index=True, return_counts=True)

    adjacency = jnp.zeros((E, E), dtype=bool)
    for i in range(unique_nodes.shape[0]):
        elems = elem_ids[node_starts[i] : node_starts[i] + node_counts[i]]
        if elems.shape[0] > 1:
            ei = elems[:, None]
            ej = elems[None, :]
            adjacency = adjacency.at[ei, ej].set(True)

    adjacency = adjacency.at[jnp.diag_indices(E)].set(False)
    adjacency = jnp.logical_or(adjacency, adjacency.T)
    return adjacency


def get_element_neighborhood(cells: jnp.ndarray, elem_index: int, depth: int = 1):
    """
    Get i-th neighborhood of a given element (JAX-compatible).

    Depth=1: direct neighbors (share node).
    Depth=2: neighbors-of-neighbors (including previous).
    """
    adjacency = build_element_adjacency(cells)
    E = adjacency.shape[0]

    visited = jnp.zeros(E, dtype=bool)
    frontier = jnp.zeros(E, dtype=bool).at[elem_index].set(True)

    for _ in range(depth):
        # neighbors of current frontier
        neighbor_mask = jnp.any(adjacency[frontier], axis=0)
        # next frontier = new neighbors not yet visited
        next_frontier = jnp.logical_and(neighbor_mask, ~visited)
        # mark current frontier as visited
        visited = jnp.logical_or(visited, frontier)
        # move to next layer
        frontier = next_frontier

    # include the last frontier as well
    visited = jnp.logical_or(visited, frontier)

    neigh_elems = jnp.where(visited)[0]
    neigh_nodes = jnp.unique(cells[neigh_elems].ravel())
    return neigh_nodes, neigh_elems
