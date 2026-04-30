"""Input/output utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_pickle(obj: Any, filename: str | Path) -> None:
    """Save a fitted model or generated result with pickle."""
    with open(filename, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str | Path) -> Any:
    """Load an object saved with :func:`save_pickle`."""
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def write_backbone_pdb_with_caps(filename, coords, chain_id="A", resname="GLY", model=None):
    """
    Write reduced capped-backbone coordinates to PDB.

    Expected coordinate layout:
        C_prev, [N_i, CA_i, C_i for i = 1..n_res], N_next
    """
    xyz = np.asarray(coords["all"], dtype=float)

    if xyz.ndim == 2:
        _write_single_backbone_pdb_with_caps(filename, xyz, chain_id=chain_id, resname=resname)

    elif xyz.ndim == 3:
        n_samples = xyz.shape[0]
        if model is not None:
            if not (0 <= model < n_samples):
                raise IndexError(f"model index {model} out of range for {n_samples} samples")
            _write_single_backbone_pdb_with_caps(
                filename, xyz[model], chain_id=chain_id, resname=resname
            )
        else:
            with open(filename, "w") as fh:
                for m in range(n_samples):
                    fh.write(f"MODEL     {m + 1:4d}\n")
                    _write_single_backbone_pdb_stream_with_caps(
                        fh, xyz[m], chain_id=chain_id, resname=resname
                    )
                    fh.write("ENDMDL\n")
                fh.write("END\n")
    else:
        raise ValueError(
            f'coords["all"] must have shape (n_atoms, 3) or '
            f"(n_samples, n_atoms, 3), got {xyz.shape}"
        )


def _write_single_backbone_pdb_with_caps(filename, xyz, chain_id="A", resname="GLY"):
    with open(filename, "w") as fh:
        _write_single_backbone_pdb_stream_with_caps(fh, xyz, chain_id=chain_id, resname=resname)
        fh.write("END\n")


def _write_single_backbone_pdb_stream_with_caps(fh, xyz, chain_id="A", resname="GLY"):
    xyz = np.asarray(xyz, dtype=float)
    n_atoms = xyz.shape[0]

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (n_atoms, 3), got {xyz.shape}")

    if (n_atoms - 2) % 3 != 0:
        raise ValueError(f"Expected n_atoms = 3*n_res + 2, got {n_atoms}")

    n_res = (n_atoms - 2) // 3
    serial = 1

    serial = _write_atom_record(
        fh, serial, "C", resname, chain_id, 0, xyz[0, 0], xyz[0, 1], xyz[0, 2], "C"
    )

    for i in range(n_res):
        resseq = i + 1
        base = 1 + 3 * i

        serial = _write_atom_record(
            fh, serial, "N", resname, chain_id, resseq,
            xyz[base, 0], xyz[base, 1], xyz[base, 2], "N"
        )
        serial = _write_atom_record(
            fh, serial, "CA", resname, chain_id, resseq,
            xyz[base + 1, 0], xyz[base + 1, 1], xyz[base + 1, 2], "C"
        )
        serial = _write_atom_record(
            fh, serial, "C", resname, chain_id, resseq,
            xyz[base + 2, 0], xyz[base + 2, 1], xyz[base + 2, 2], "C"
        )

    _write_atom_record(
        fh, serial, "N", resname, chain_id, n_res + 1,
        xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], "N"
    )


def _write_atom_record(fh, serial, atom_name, resname, chain_id, resseq, x, y, z, element):
    line = (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain_id:1s}"
        f"{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )
    fh.write(line)
    return serial + 1
