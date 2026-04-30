import numpy as np

from backbonegpm.builder import build_backbone_with_local_geometry, internal_coords_from_reduced_backbone


def test_builder_single_smoke():
    phi_psi = np.array([[-60.0, -40.0], [-70.0, 130.0]])
    coords = build_backbone_with_local_geometry(phi_psi)
    assert coords["all"].shape == (8, 3)

    df = internal_coords_from_reduced_backbone(coords)
    assert len(df) == 2
