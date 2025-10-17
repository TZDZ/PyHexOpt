from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def nodes_from_points(points: jax.Array, cells: jax.Array) -> jax.Array:
    """
    Gather per-element node coordinates.
    points: (N,3)
    cells:  (E,8)
    returns node_coords: (E,8,3)
    """
    return points[cells]  # (E,8,3)


@jax.jit
def apply_nodal_displacements(points: jax.Array, disp: jax.Array) -> jax.Array:
    """Add disp (N,3) to points (N,3), returns new points."""
    return points + disp


@partial(jax.jit, static_argnames=("mode", "N_nodes"))
def reduce_element_deltas_to_nodal(
    cells: jax.Array,
    dnode_coords: jax.Array,
    N_nodes: int,
    mode: str = "average",
):
    """
    Convert per-element-per-corner deltas to per-node aggregated deltas.

    Args:
        cells: (E,8) int array of node indices.
        dnode_coords: (E,8,3) per-element deltas for each corner.
        N_nodes: int, number of global nodes.
        mode: str, either 'average' or 'sum'.
            'average' divides the aggregated sum by the number of contributions per node.
            'sum' returns the summed delta.

    Returns:
        nodal_delta: (N_nodes, 3) array of aggregated nodal deltas.

    """
    # flatten
    E = cells.shape[0]
    idx_flat = jnp.reshape(cells, (E * 8,))  # (E*8,)
    deltas_flat = jnp.reshape(dnode_coords, (E * 8, 3))  # (E*8,3)

    # segment_sum: sums deltas by node index
    sums = jax.ops.segment_sum(deltas_flat, idx_flat, N_nodes)  # (N_nodes, 3)

    if mode == "sum":
        return sums

    # average: need counts per node
    ones = jnp.ones((idx_flat.shape[0],), dtype=sums.dtype)
    counts = jax.ops.segment_sum(ones, idx_flat, N_nodes)  # (N_nodes,)
    # avoid divide-by-zero: where counts == 0 -> keep 0 delta
    counts = jnp.maximum(counts, 1.0)
    avg = sums / counts.reshape((-1, 1))
    return avg


@jax.jit
def update_points_from_element_deltas(
    points: jax.Array,
    cells: jax.Array,
    dnode_coords: jax.Array,
    mode: str = "average",
):
    """
    Apply per-element deltas to global points, aggregating duplicates.

    Returns:
      new_points: (N,3)

    """
    N = points.shape[0]
    nodal_delta = reduce_element_deltas_to_nodal(cells, dnode_coords, N, mode=mode)
    return points + nodal_delta


@partial(jax.jit, static_argnames=("N"))
def uv_to_disp_full(free_uv: jax.Array, T1: jax.Array, T2: jax.Array, movable_idx: jax.Array, N: int):
    """
    free_uv: (M,2)
    T1, T2: (M,3)
    movable_idx: (M,) int32
    Returns: disp_full (N,3)
    """
    # compute free displacements in xyz
    # free_uv[:,0] -> shape (M,), broadcast to (M,1)
    disp_free = free_uv[:, 0:1] * T1 + free_uv[:, 1:2] * T2  # (M,3)

    disp_full = jnp.zeros((N, 3), dtype=disp_free.dtype)
    disp_full = disp_full.at[movable_idx].set(disp_free)
    return disp_full
