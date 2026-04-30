"""Feature extraction utilities for macrostate shapeGMM fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_universe(topology: str | Path, trajectory: str | Path | Sequence[str | Path] | None = None):
    """Load an MDAnalysis Universe lazily so MDAnalysis can remain an optional dependency."""
    import MDAnalysis as mda

    if trajectory is None:
        return mda.Universe(str(topology))
    return mda.Universe(str(topology), trajectory)


def trajectory_atom_positions(universe: Any, selection: str = "name N CA C") -> np.ndarray:
    """
    Return selected atom positions over a trajectory.

    Parameters
    ----------
    universe
        MDAnalysis Universe.
    selection
        MDAnalysis atom selection string.

    Returns
    -------
    positions
        Array with shape ``(n_frames, n_selected_atoms, 3)``.
    """
    ag = universe.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"Selection produced no atoms: {selection!r}")

    frames = []
    for _ts in universe.trajectory:
        frames.append(ag.positions.copy())

    return np.asarray(frames, dtype=float)


def center_positions(positions: np.ndarray) -> np.ndarray:
    """
    Center each frame by subtracting its centroid.

    This is useful if the shapeGMM input should be translation-free. Use only if
    your shapeGMM workflow does not already center/superpose internally.
    """
    positions = np.asarray(positions, dtype=float)
    return positions - positions.mean(axis=1, keepdims=True)
