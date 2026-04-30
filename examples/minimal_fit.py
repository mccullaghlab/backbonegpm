"""
Minimal example. This assumes shapeGMMTorch, multi, plgpm, and MDAnalysis are installed.
"""

import numpy as np

from backbonegpm import FitConfig, HierarchicalBackboneGPM

components = [
    np.array([1, 2], dtype=int),
    np.array([2, 2], dtype=int),
]

config = FitConfig(
    n_macrostates=2,
    microstate_components=components,
    macro_selection="name N CA C",
    internal_selection="resname ACE ALA NME",
    delta_fit=10,
)

model = HierarchicalBackboneGPM.from_files(
    topology="topology.prmtop",
    trajectory=["trajectory.nc"],
    config=config,
)

# model.fit()
# generated = model.generate(n_samples=1000)
