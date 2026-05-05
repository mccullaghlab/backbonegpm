import numpy as np

from backbonegpm.model import FitConfig, HierarchicalBackboneGPM


class DummyMacro:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class DummyBV:
    def __init__(self):
        self.components = np.array([2, 2])

    def fit(self, *args, **kwargs):
        return self

    def refine(self, *args, **kwargs):
        return self

    def predict_micro(self, X):
        return np.zeros(len(X), dtype=int)


class DummyGPM:
    def sample(self, n_samples):
        return np.zeros((n_samples, 2), dtype=int)


class FastModel(HierarchicalBackboneGPM):
    def _fit_shape_gmm(self, X):
        return DummyMacro()

    def _fit_bvvmmm(self, phi_psi_m, components_m):
        return DummyBV()

    def _fit_gpm(self, z_m, components):
        return DummyGPM()


def test_microstate_scan_and_fit_pipeline(monkeypatch):
    cfg = FitConfig(n_macrostates=1, microstate_components=[np.array([1, 1])], fit_local_geometry=False)
    model = FastModel(cfg)
    model.macro_features_ = np.random.randn(8, 4, 3)
    model.phi_psi_ = np.random.randn(8, 3, 2)
    model.internal_df_ = None
    model.model_resids_ = [1, 2, 3]
    model.fit_macrostates()

    class ScanBV:
        def __init__(self):
            self.components = np.array([3, 2, 2])

        def component_scan(self, X):
            return self

    import types
    import sys
    mod = types.SimpleNamespace(MultiIndSineBVvMMM=ScanBV)
    monkeypatch.setitem(sys.modules, "multi", mod)

    model.scan_and_fit_microstates()

    assert model.config.microstate_components[0].tolist() == [3, 2, 2]
    assert len(model.macrostates_) == 1
    assert model.is_fitted_
