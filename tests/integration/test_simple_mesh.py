from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements
from pyhexopt.core.obj import expand_disp_from_mask, expand_displacements, objective_free
from pyhexopt.core.utils import get_boundary_nodes, prepare_dof_masks_and_bases
from pyhexopt.main import main, main_simple


def test_real_mesh_masked_grad(clean_square_mesh, out_path: Path):
    ### data du test
    move = np.array([0.1, 0.2, 0.3])
    moving_node = 59  # indexings starts at 0!

    ### Lecture du maillage
    msh = clean_square_mesh
    initpos = np.array(msh.points[moving_node])

    ### préproc
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    boundary = get_boundary_nodes(points, cells)
    disp = jnp.zeros_like(points)  # shape (N,3)
    disp = disp.at[moving_node].set(jnp.array(move, dtype=jnp.float32))
    fixed_indices = jnp.array(boundary)  # e.g., one face is fixed
    free_mask = jnp.ones((points.shape[0],), dtype=bool).at[fixed_indices].set(False)
    free_disp0 = disp[free_mask]

    ### Pour l'optim : objective_free -> sortira free_disp_opt
    assert objective_free(free_disp0, points, cells, free_mask) > 0
    grad = jax.grad(objective_free)(free_disp0, points, cells, free_mask)
    np.testing.assert_allclose(grad[0], 0.0)
    assert np.any(np.abs(grad[1]) > 0.01)
    assert grad.shape[0] == len(points) - len(boundary)

    ###test : est-ce qu'on reconstruit bien le maillage déplacé ?
    disp_ = expand_disp_from_mask(free_disp0, free_mask)
    new_points = apply_nodal_displacements(points, disp_)
    new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
    out_file = out_path / "test_simple_mesh_out.msh"
    if out_file.exists():
        out_file.unlink()
    new_mesh.write(out_file, file_format="gmsh")
    msh = meshio.read(out_file)
    np.testing.assert_allclose(np.array(msh.points[moving_node]), move + initpos)


def test_end_to_end_simple(clean_square_mesh, square_bad1_mesh, out_path):
    out_mesh_path = out_path / "corrected_simple_mesh.msh"
    main_simple(square_bad1_mesh, out_mesh_path)
    corrected_msh = meshio.read(out_mesh_path)
    np.testing.assert_allclose(clean_square_mesh.points, corrected_msh.points, atol=2e-3)


def test_end_to_end(clean_square_mesh, square_bad1_mesh, out_path):
    out_mesh_path = out_path / "corrected_simple_mesh.msh"
    main(square_bad1_mesh, out_mesh_path)
    corrected_msh = meshio.read(out_mesh_path)
    np.testing.assert_allclose(clean_square_mesh.points, corrected_msh.points, atol=2e-3)


@pytest.mark.parametrize("mesh_name", ["square_rot_bad1_mesh", "square_rot_bad2_mesh", "square_rot_bad3_mesh"])
def test_end_to_end_rot(clean_rot_square_mesh, mesh_name, out_path, request):
    mesh_fixture = request.getfixturevalue(mesh_name)
    out_mesh_path = out_path / "corrected_simple_rot_mesh.msh"
    main(mesh_fixture, out_mesh_path)
    corrected_msh = meshio.read(out_mesh_path)
    np.testing.assert_allclose(clean_rot_square_mesh.points, corrected_msh.points, atol=2e-3)


@pytest.mark.parametrize("mesh_name", ["clean_square_mesh", "clean_rot_square_mesh"])
def test_move_mode_surface(mesh_name, request, out_path):
    mesh_fixture = request.getfixturevalue(mesh_name)
    points, cells = extract_points_and_cells(mesh_fixture, dtype=jnp.float32)
    volumic_nodes, surface_nodes, edge_nodes, T1, T2 = prepare_dof_masks_and_bases(points, cells)

    n_tot = points.shape[0]
    is_free = np.zeros(n_tot, dtype=bool)
    is_free[volumic_nodes] = True

    is_surface = np.zeros(n_tot, dtype=bool)
    is_surface[surface_nodes] = True

    n_volu = len(volumic_nodes)
    n_surf = len(surface_nodes)
    reduced_disps = jnp.concatenate(
        [
            jnp.zeros((n_volu * 3,)),  # flatten free 3D displacements
            jnp.zeros((len(surface_nodes) * 2,)),  # flatten surface 2D displacements
        ]
    )

    reduced_disps = reduced_disps.at[n_volu * 3 + 10].set(0.25)

    volu_disps = reduced_disps[: n_volu * 3].reshape((n_volu, 3))
    surf_disps = reduced_disps[n_volu * 3 :].reshape((n_surf, 2))

    all_disps = expand_displacements(volu_disps, surf_disps, volumic_nodes, surface_nodes, T1, T2, n_tot)

    new_points = apply_nodal_displacements(points, all_disps)

    new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
    mesh_out = out_path / f"surf_move_{mesh_name}.msh"
    if mesh_out.exists():
        mesh_out.unlink()
    new_mesh.write(mesh_out, file_format="gmsh")
    moved_mesh = meshio.read(mesh_out)

    orig_pts = np.array(mesh_fixture.points)
    new_pts = np.array(moved_mesh.points)
    diff = np.linalg.norm(new_pts - orig_pts, axis=1)
    moved_mask = diff > 1e-6  # tolerance for floating-point noise
    moved_indices = np.nonzero(moved_mask)[0]
    assert len(moved_indices) == 1
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_allclose(orig_pts, new_pts, atol=1e-6)


@pytest.mark.parametrize("mesh_name", ["clean_square_mesh", "clean_rot_square_mesh"])
def test_surface_displacement_affects_only_surface_nodes(mesh_name, request, out_path):
    mesh_fixture = request.getfixturevalue(mesh_name)
    points, cells = extract_points_and_cells(mesh_fixture, dtype=jnp.float32)
    volumic_nodes, surface_nodes, edge_nodes, T1, T2 = prepare_dof_masks_and_bases(points, cells)

    n_tot = points.shape[0]
    n_volu = len(volumic_nodes)
    n_surf = len(surface_nodes)

    # Initialize reduced displacements (all zeros)
    reduced_disps = jnp.concatenate(
        [
            jnp.zeros((n_volu * 3,)),  # free volumic displacements (x,y,z)
            jnp.zeros((n_surf * 2,)),  # surface displacements (u,v)
        ]
    )

    # --- Apply a surface displacement modification ---
    # For example: add some displacement in u-direction of surface node 5
    surface_disp_index = n_volu * 3 + 5 * 2  # u-component of 6th surface node
    reduced_disps = reduced_disps.at[surface_disp_index].set(0.3)

    # --- Expand displacements into full 3D displacements ---
    volu_disps = reduced_disps[: n_volu * 3].reshape((n_volu, 3))
    surf_disps = reduced_disps[n_volu * 3 :].reshape((n_surf, 2))

    all_disps = expand_displacements(volu_disps, surf_disps, volumic_nodes, surface_nodes, T1, T2, n_tot)

    # --- Apply displacements ---
    moved_points = apply_nodal_displacements(points, all_disps)
    orig_pts = np.array(points)
    new_pts = np.array(moved_points)

    # --- Compute per-node displacement magnitudes ---
    diff = np.linalg.norm(new_pts - orig_pts, axis=1)

    # --- Verify: only surface nodes moved ---
    tol = 1e-8
    moved_mask = diff > tol

    moved_indices = np.nonzero(moved_mask)[0]

    # (1) All moved indices are a subset of surface nodes
    assert set(moved_indices).issubset(set(np.array(surface_nodes)))

    # (2) Volumic nodes remained fixed
    assert np.allclose(
        new_pts[np.array(volumic_nodes)],
        orig_pts[np.array(volumic_nodes)],
        atol=tol,
    )

    # (3) At least one surface node moved (sanity check)
    assert len(moved_indices) > 0

    # Optionally, write mesh for inspection
    mesh_out = out_path / f"surf_move_{mesh_name}.msh"
    if mesh_out.exists():
        mesh_out.unlink()
    new_mesh = meshio.Mesh(points=np.array(new_pts), cells=[("hexahedron", np.array(cells))])
    new_mesh.write(mesh_out, file_format="gmsh")


if __name__ == "__main__":
    pytest.main([__file__])
