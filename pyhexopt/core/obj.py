import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from pyhexopt.core.jaxobian import GAUSS_POINTS, compute_scaled_jacobians_from_coords
from pyhexopt.core.move import nodes_from_points, uv_to_disp_full


def apply_masked_displacement(points: ArrayLike, disp: ArrayLike, free_mask: ArrayLike) -> jax.Array:
    """Apply displacements only to free nodes (JAX-friendly)."""
    mask = free_mask.astype(points.dtype)[:, None]
    return points + disp * mask


def objective(disp: ArrayLike, points: ArrayLike, cells: ArrayLike, free_mask: ArrayLike) -> jax.Array:
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


# @jax.jit
def expand_displacements(
    free_disp_3d,
    surface_disp_uv,
    free_nodes,
    surface_nodes,
    T1: np.ndarray,
    T2: np.ndarray,
    N: int,
):
    """
    Reconstruct full (N,3) displacement array using:
      - free_disp_3d: (F,3) moves for fully-free nodes
      - surface_disp_uv: (S,2) local u,v moves for surface nodes
      - free_nodes: array-like (F,) indices
      - surface_nodes: array-like (S,) indices
      - T1,T2: (S,3) tangent bases corresponding to surface_nodes
      - N: total number of nodes

    This version is numpy-based (callable from tests). For JAX use uv_to_disp_full jitted helper.
    """
    free_nodes = np.asarray(free_nodes, dtype=int)
    surface_nodes = np.asarray(surface_nodes, dtype=int)
    free_disp_3d = np.asarray(free_disp_3d, dtype=float)
    surface_disp_uv = np.asarray(surface_disp_uv, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    T2 = np.asarray(T2, dtype=float)

    disp_full = np.zeros((N, 3), dtype=float)

    if free_nodes.size > 0:
        assert free_disp_3d.shape[0] == free_nodes.shape[0]
        disp_full[free_nodes] = free_disp_3d

    if surface_nodes.size > 0:
        assert surface_disp_uv.shape[0] == surface_nodes.shape[0]
        # convert uv -> xyz for surface nodes
        disp_surface = surface_disp_uv[:, 0:1] * T1 + surface_disp_uv[:, 1:2] * T2
        disp_full[surface_nodes] = disp_surface

    return disp_full


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


# Example objective that mirrors your objective_free but uses free_uv (M,2)
@jax.jit
def objective_uv(free_uv, points, cells, T1, T2, movable_idx):
    N = points.shape[0]
    disp_full = uv_to_disp_full(free_uv, T1, T2, movable_idx, N)
    moved_points = points + disp_full

    # per-element coordinates (keep nodes_from_points and jac fns JAX-friendly)
    node_coords = nodes_from_points(moved_points, cells)

    jac = compute_scaled_jacobians_from_coords(node_coords, at_center=False, sample_points=GAUSS_POINTS)
    min_jac = jnp.min(jac, axis=1)
    return jnp.sum((min_jac - 1.0) ** 2)


def objective_mixed_dof(
    disp_concat,  # concatenated array: [free_disp_3d, surface_disp_uv]
    points,
    cells,
    free_nodes,
    surface_nodes,
    T1,
    T2,
):
    n_free = len(free_nodes)
    free_disp_3d = disp_concat[:n_free]
    surface_disp_uv = disp_concat[n_free:]

    disp_full = expand_displacements(free_disp_3d, surface_disp_uv, free_nodes, surface_nodes, T1, T2, points.shape[0])

    moved_points = points + disp_full
    node_coords = nodes_from_points(moved_points, cells)
    jac = compute_scaled_jacobians_from_coords(node_coords, at_center=False, sample_points=GAUSS_POINTS)
    min_jac = jnp.min(jac, axis=1)
    return jnp.sum((min_jac - 1.0) ** 2)
