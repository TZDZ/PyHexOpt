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


def get_element_neighborhood_old(cells: jnp.ndarray, elem_index: int, depth: int = 1):
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


def build_node_to_elements_sorted(cells: jnp.ndarray):
    """
    Build arrays:
      node_ids_sorted : shape (8*E,) sorted node indices (repeated per element)
      elem_ids_sorted : shape (8*E,) element indices aligned with node_ids_sorted
    plus unique_nodes, node_starts, node_counts (optional).
    """
    E, Nn = cells.shape
    elem_ids = jnp.repeat(jnp.arange(E, dtype=jnp.int32), Nn)  # (8*E,)
    node_ids = cells.ravel()  # (8*E,)
    order = jnp.argsort(node_ids)
    node_ids_sorted = node_ids[order]
    elem_ids_sorted = elem_ids[order]

    # unique_nodes, node_starts, node_counts may be useful for other uses
    unique_nodes, node_starts, node_counts = jnp.unique(node_ids_sorted, return_index=True, return_counts=True)

    return {
        "node_ids_sorted": node_ids_sorted,
        "elem_ids_sorted": elem_ids_sorted,
        "unique_nodes": unique_nodes,
        "node_starts": node_starts,
        "node_counts": node_counts,
    }


def get_element_neighborhood(
    cells: jnp.ndarray,
    node_elem_index_data: dict,
    elem_index: int,
    depth: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get nodes and elements in the depth-neighborhood of elem_index using
    node->elements sorted arrays (JAX-compatible).
    Returns (neigh_nodes, neigh_elems).
    """
    E = cells.shape[0]
    node_ids_sorted = node_elem_index_data["node_ids_sorted"]
    elem_ids_sorted = node_elem_index_data["elem_ids_sorted"]

    visited = jnp.zeros(E, dtype=bool)
    frontier = jnp.zeros(E, dtype=bool).at[elem_index].set(True)

    for _ in range(depth):
        # elements currently on the frontier (exact list, no padding)
        frontier_elems = jnp.where(frontier)[0]  # variable-length array of indices

        # if frontier is empty, stop
        if frontier_elems.size == 0:
            break

        # gather nodes used by frontier elements
        nodes = jnp.unique(cells[frontier_elems].ravel())

        # find all element ids that reference any of these nodes
        mask = jnp.isin(node_ids_sorted, nodes)
        elems_touching_nodes = jnp.unique(elem_ids_sorted[mask])

        # mark current frontier visited first
        visited = jnp.logical_or(visited, frontier)

        # new frontier = those elems touching nodes but not yet visited
        next_frontier_mask = jnp.logical_and(
            jnp.zeros(E, dtype=bool).at[elems_touching_nodes].set(True), jnp.logical_not(visited)
        )
        frontier = next_frontier_mask

    # include last frontier
    visited = jnp.logical_or(visited, frontier)

    neigh_elems = jnp.where(visited)[0]
    neigh_nodes = jnp.unique(cells[neigh_elems].ravel())
    return neigh_nodes, neigh_elems
