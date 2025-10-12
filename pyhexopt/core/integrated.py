import jax.numpy as jnp
import meshio

from pyhexopt.adapters.meshio_ import extract_hex_node_coords
from pyhexopt.core.jaxobian import compute_scaled_jacobians_from_coords


def compute_scaled_jacobians(
    mesh: meshio.Mesh,
    dtype=jnp.float32,
    at_center=True,
    sample_points=None,
    eps=0.0,
):
    """
    Compute scaled Jacobians directly from a meshio mesh.

    Wrapper that extracts node coordinates and calls
    `compute_scaled_jacobians_from_coords`.
    """
    node_coords = extract_hex_node_coords(mesh, dtype=dtype)
    return compute_scaled_jacobians_from_coords(
        node_coords=node_coords,
        dtype=dtype,
        at_center=at_center,
        sample_points=sample_points,
        eps=eps,
    )
