from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from pyhexopt.core.jaxobian import GAUSS_POINTS, compute_scaled_jacobians_from_coords, hex_edge_aspect_ratio
from pyhexopt.core.move import apply_nodal_displacements, nodes_from_points


def apply_masked_displacement(points: ArrayLike, disp: ArrayLike, free_mask: ArrayLike) -> jax.Array:
    """Apply displacements only to free nodes (JAX-friendly)."""
    mask = free_mask.astype(points.dtype)[:, None]
    return points + disp * mask


@jax.jit
def expand_disp_from_mask(free_disp: ArrayLike, free_mask: ArrayLike) -> jax.Array:
    """Reconstruct full displacement from free node displacements and a boolean free_mask."""
    N = free_mask.shape[0]
    full = jnp.zeros((N, 3), dtype=free_disp.dtype)
    free_indices = jnp.where(free_mask, size=free_disp.shape[0])[0]
    full = full.at[free_indices].set(free_disp)
    return full


@partial(jax.jit, static_argnames=("N",))
def expand_displacements(  # noqa: PLR0913
    free_disp_3d: jax.Array,
    surface_disp_uv: jax.Array,
    free_nodes: jax.Array,
    surface_nodes: jax.Array,
    T1: jax.Array,
    T2: jax.Array,
    N: int,
) -> jax.Array:
    """
    JAX-compatible version of expand_displacements.

    Args:
        free_disp_3d: (F,3) displacement vectors for fully-free nodes
        surface_disp_uv: (S,2) local UV displacements for surface nodes
        free_nodes: (F,) indices of fully-free nodes
        surface_nodes: (S,) indices of surface nodes
        T1: (S,3) tangent bases for surface nodes
        T2: (S,3) tangent bases for surface nodes
        N: total number of nodes

    Returns:
        disp_full: (N,3) displacement field for all nodes

    """
    # ensure JAX arrays (safe if already are)
    free_nodes = jnp.asarray(free_nodes, dtype=jnp.int32)
    surface_nodes = jnp.asarray(surface_nodes, dtype=jnp.int32)
    free_disp_3d = jnp.asarray(free_disp_3d, dtype=jnp.float32)
    surface_disp_uv = jnp.asarray(surface_disp_uv, dtype=jnp.float32)
    T1 = jnp.asarray(T1, dtype=jnp.float32)
    T2 = jnp.asarray(T2, dtype=jnp.float32)

    disp_full = jnp.zeros((N, 3), dtype=jnp.float32)

    # Scatter free node displacements
    disp_full = disp_full.at[free_nodes].set(free_disp_3d)

    # Compute and scatter surface node displacements (local UV → XYZ)
    if surface_nodes.size > 0:
        disp_surface = surface_disp_uv[:, 0:1] * T1 + surface_disp_uv[:, 1:2] * T2
        disp_full = disp_full.at[surface_nodes].set(disp_surface)

    return disp_full


@jax.jit
def objective_simple(
    free_disp: ArrayLike,
    points: ArrayLike,
    cells: ArrayLike,
    free_mask: ArrayLike,
) -> jax.Array:
    """
    Objective function where only a subset of nodes (free_mask) are allowed to move.

    Parameters
    ----------
    free_disp : (F,3)
        Displacements for the free nodes only.
    points : (N,3)
        Original coordinates.
    cells : (E,8)
        Connectivity.
    free_mask : (N,)
        Boolean mask, True for free nodes, False for fixed nodes.

    Returns
    -------
    obj : scalar
        Objective value, e.g. deviation of Jacobian determinant from 1.

    """
    disp = expand_disp_from_mask(free_disp, free_mask)
    moved_points = points + disp
    node_coords = nodes_from_points(moved_points, cells)

    jac = compute_scaled_jacobians_from_coords(
        node_coords,
        at_center=False,
        sample_points=GAUSS_POINTS,
    )  # (E, 8)

    min_jac = jnp.min(jac, axis=1)
    return jnp.sum((min_jac - 1.0) ** 2)


@partial(jax.jit, static_argnames=("n_volu", "n_surf", "n_tot", "alpha"))
def objective(  # noqa: PLR0913
    reduced_disps: jax.Array,
    points: jax.Array,
    cells: jax.Array,
    n_volu: int,
    n_surf: int,
    n_tot: int,
    volumic_nodes: jax.Array,
    surface_nodes: jax.Array,
    T1: jax.Array,
    T2: jax.Array,
    alpha: float = 1.0,
) -> jax.Array:
    volu_disps = reduced_disps[: n_volu * 3].reshape((n_volu, 3))
    surf_disps = reduced_disps[n_volu * 3 :].reshape((n_surf, 2))

    all_disps = expand_displacements(volu_disps, surf_disps, volumic_nodes, surface_nodes, T1, T2, n_tot)

    moved_points = apply_nodal_displacements(points, all_disps)
    node_coords = nodes_from_points(moved_points, cells)
    jac = compute_scaled_jacobians_from_coords(node_coords, at_center=False, sample_points=GAUSS_POINTS)
    worst_jac = jnp.min(jac, axis=1)
    aspect_ratio = hex_edge_aspect_ratio(node_coords)
    score = jnp.abs(worst_jac - 1.0) ** (alpha) * aspect_ratio ** (1 - alpha)
    return jnp.sum(score**2)
