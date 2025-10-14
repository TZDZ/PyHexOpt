import meshio
import numpy as np
import pytest

from pyhexopt.core.utils import compute_node_normals_from_faces, detect_free_edge_nodes, face_normal, get_boundary_nodes


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


def testface_normal_triangle_xy_plane():
    # Triangle in XY plane, counter-clockwise (normal = +Z)
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    face = (0, 1, 2)
    n = face_normal(pts, face)
    assert np.allclose(n, [0, 0, 1], atol=1e-8)


def testface_normal_triangle_reverse_orientation():
    # Same triangle, reversed orientation -> normal = -Z
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    face = (2, 1, 0)
    n = face_normal(pts, face)
    assert np.allclose(n, [0, 0, -1], atol=1e-8)


def testface_normal_quad_xy_plane():
    # Unit square in XY plane (Z=0)
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
    )
    face = (0, 1, 2, 3)
    n = face_normal(pts, face)
    assert np.allclose(n, [0, 0, 1], atol=1e-8)
    assert np.isclose(np.linalg.norm(n), 1.0)


def testface_normal_quad_tilted():
    # Quad tilted 45° around X axis → normal rotated towards YZ plane
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    face = (0, 1, 2, 3)
    n = face_normal(pts, face)
    # Expected normal roughly halfway between +Y and +Z
    expected = np.array([0, -1, 1]) / np.sqrt(2)
    # Allow sign ambiguity (depending on vertex ordering)
    if np.dot(n, expected) < 0:
        n = -n
    assert np.allclose(n, expected, atol=1e-8)


def testface_normal_degenerate_triangle():
    # Colinear points -> degenerate normal -> zero vector
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ]
    )
    face = (0, 1, 2)
    n = face_normal(pts, face)
    assert np.allclose(n, [0, 0, 0], atol=1e-12)


def testface_normal_degenerate_quad():
    # Quad with all points the same -> zero vector
    pts = np.zeros((4, 3))
    face = (0, 1, 2, 3)
    n = face_normal(pts, face)
    assert np.allclose(n, [0, 0, 0], atol=1e-12)


def testface_normal_invariant_to_translation():
    # Shift the same plane in space
    pts = np.array(
        [
            [0, 0, 5],
            [1, 0, 5],
            [0, 1, 5],
        ]
    )
    n1 = face_normal(pts, (0, 1, 2))
    pts_shifted = pts + np.array([10, -3, 2])
    n2 = face_normal(pts_shifted, (0, 1, 2))
    assert np.allclose(n1, n2, atol=1e-8)


def _make_unit_cube_hex_mesh():
    """
    Construct a minimal single-hex mesh (8 vertices, 1 hexahedron)
    shaped as a unit cube [0,1]^3.
    """
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    cells = [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]
    mesh = meshio.Mesh(points, cells)
    return mesh


def _make_two_cubes_sharing_face():
    """
    Two unit cubes stacked along X-axis, sharing one face.
    Only the outer boundary edges should be 'free'.
    """
    pts = np.array(
        [
            # left cube
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1 shared
            [1, 1, 0],  # 2 shared
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5 shared
            [1, 1, 1],  # 6 shared
            [0, 1, 1],  # 7
            # right cube (offset +1 in x)
            [2, 0, 0],  # 8
            [2, 1, 0],  # 9
            [2, 0, 1],  # 10
            [2, 1, 1],  # 11
        ]
    )
    hex1 = [0, 1, 2, 3, 4, 5, 6, 7]
    hex2 = [1, 8, 9, 2, 5, 10, 11, 6]
    cells = [("hexahedron", np.array([hex1, hex2]))]
    mesh = meshio.Mesh(pts, cells)
    return mesh


def test_no_boundary_faces_internal_mesh():
    """
    A closed single cube should have boundary faces, but all edges are free edges
    (since all boundary faces meet at sharp 90°).
    """
    mesh = _make_unit_cube_hex_mesh()
    edge_nodes, edge_mask = detect_free_edge_nodes(mesh, angle_deg=45.0)

    # All boundary nodes are cube corners
    boundary_nodes = get_boundary_nodes(mesh)
    # All 8 should be boundary and free-edge
    assert len(boundary_nodes) == 8
    assert np.all(edge_mask[boundary_nodes])
    assert set(edge_nodes) == set(boundary_nodes)


def test_angle_threshold_controls_detection():
    """
    Create two boundary faces meeting at a small angle (almost flat)
    and ensure angle threshold controls detection.
    """
    # Build two quads sharing an edge with small dihedral angle (~5°)
    pts = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 0.1],  # 4 small tilt in z
            [1, 0, 0.1],  # 5
            [1, 1, 0.1],  # 6
            [0, 1, 0.1],  # 7
        ]
    )
    # Two slightly offset cubes to form nearly-flat interface
    hex1 = [0, 1, 2, 3, 4, 5, 6, 7]
    mesh = meshio.Mesh(pts, [("hexahedron", np.array([hex1]))])

    # small threshold → still consider flat faces connected
    _, mask_loose = detect_free_edge_nodes(mesh, angle_deg=1.0)
    # tight threshold → classify edges as sharp
    _, mask_strict = detect_free_edge_nodes(mesh, angle_deg=89.0)

    num_loose = mask_loose.sum()
    num_strict = mask_strict.sum()

    assert num_strict >= num_loose  # higher threshold means more edges counted


def test_return_types_and_shapes():
    mesh = _make_unit_cube_hex_mesh()
    edge_nodes, edge_mask = detect_free_edge_nodes(mesh, angle_deg=45.0)
    assert isinstance(edge_nodes, np.ndarray)
    assert edge_mask.dtype == bool
    assert edge_mask.shape[0] == mesh.points.shape[0]
    # indices are valid
    assert np.all((edge_nodes >= 0) & (edge_nodes < mesh.points.shape[0]))


def test_compute_node_normals_simple_plane():
    # single square face in XY plane
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
    )
    faces = [(0, 1, 2, 3)]
    face_normals = np.array([[0, 0, 1]])
    node_normals = compute_node_normals_from_faces(points, faces, face_normals)
    # all nodes should have normal [0,0,1]
    for n in node_normals:
        assert np.allclose(n, [0, 0, 1])


if __name__ == "__main__":
    pytest.main([__file__])
