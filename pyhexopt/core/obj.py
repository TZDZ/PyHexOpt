import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from pyhexopt.core.jaxobian import GAUSS_POINTS, compute_scaled_jacobians_from_coords
from pyhexopt.core.move import nodes_from_points


def apply_masked_displacement(points: ArrayLike, disp: ArrayLike, free_mask: ArrayLike) -> jax.Array:
    """Apply displacements only to free nodes (JAX-friendly)."""
    mask = free_mask.astype(points.dtype)[:, None]
    return points + disp * mask


def objective(disp: ArrayLike, points: ArrayLike, cells: ArrayLike, free_mask: ArrayLike):
    """Example: minimize average deviation of Jacobian determinant from 1."""
    moved_points = apply_masked_displacement(points, disp, free_mask)
    node_coords = nodes_from_points(moved_points, cells)

    jac = compute_scaled_jacobians_from_coords(node_coords, at_center=False, sample_points=GAUSS_POINTS)
    worst_jac = jnp.min(jac, axis=1)
    return jnp.sum((worst_jac - 1.0) ** 2)


@jax.jit
def expand_disp_from_mask(free_disp: ArrayLike, free_mask: ArrayLike) -> jax.Array:
    """Reconstruct full displacement from free node displacements and a boolean free_mask."""
    N = free_mask.shape[0]
    full = jnp.zeros((N, 3), dtype=free_disp.dtype)
    free_indices = jnp.where(free_mask, size=free_disp.shape[0])[0]
    full = full.at[free_indices].set(free_disp)
    return full


@jax.jit
def objective_free(
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
    # Expand displacements to full node set
    disp = expand_disp_from_mask(free_disp, free_mask)

    # Apply the displacements
    moved_points = points + disp

    # Per-element coordinates
    node_coords = nodes_from_points(moved_points, cells)

    # Compute per-element Jacobians
    jac = compute_scaled_jacobians_from_coords(
        node_coords,
        at_center=False,
        sample_points=GAUSS_POINTS,
    )  # (E, 8)

    # Example objective: penalize deviation of smallest jacobian from 1
    min_jac = jnp.min(jac, axis=1)
    return jnp.sum((min_jac - 1.0) ** 2)
