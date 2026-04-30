# backbonegpm

Prototype package for a hierarchical peptide-backbone model:

```text
shapeGMM macrostates -> BVVMMM residue microstates -> GPM residue coupling -> generated backbones
```

## Development install

From the repository root:

```bash
python -m pip install -e ".[md,torch,dev]"
```

If `shapeGMMTorch`, `multi`, or `plgpm` are local packages, install them separately in the same environment.

## Minimal usage

```python
import numpy as np
from backbonegpm import FitConfig, HierarchicalBackboneGPM

components = [
    np.array([1, 2]),
    np.array([2, 2]),
]

config = FitConfig(
    n_macrostates=2,
    microstate_components=components,
    macro_selection="name N CA C",
    internal_selection="resname ACE ALA NME",
)

model = HierarchicalBackboneGPM.from_files("topology.prmtop", ["traj.nc"], config)
model.fit()
samples = model.generate(n_samples=1000)
```

## Current status

This is an early package skeleton. The core objects are in `model.py`; internal-coordinate extraction is in `internal.py`; local-geometry emissions are in `emissions.py`; coordinate generation is in `builder.py`; trajectory feature extraction is in `features.py`; and PDB writing is in `io.py`.
