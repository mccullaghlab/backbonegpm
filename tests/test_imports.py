def test_public_imports():
    import backbonegpm

    assert hasattr(backbonegpm, "FitConfig")
    assert hasattr(backbonegpm, "HierarchicalBackboneGPM")
