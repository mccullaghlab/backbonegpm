"""Small residue/selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_model_resids(internal_df: pd.DataFrame) -> list[int]:
    """Residues modeled by BVVMMM/GPM are rows with valid phi and psi."""
    required = {"resid", "phi", "psi"}
    missing = required - set(internal_df.columns)
    if missing:
        raise ValueError(f"internal_df is missing required columns: {sorted(missing)}")

    resids = internal_df.loc[
        internal_df["phi"].notna() & internal_df["psi"].notna(), "resid"
    ].unique()
    return sorted(int(x) for x in resids)


def extract_phi_psi_from_internal_df(
    internal_df: pd.DataFrame,
    model_resids: list[int] | None = None,
    angle_unit: str = "radians",
) -> tuple[np.ndarray, list[int]]:
    """
    Convert an internal-coordinate dataframe into an array of phi/psi values.

    Returns shape ``(n_frames, n_model_residues, 2)``.
    """
    if model_resids is None:
        model_resids = infer_model_resids(internal_df)

    frames = sorted(internal_df["frame"].unique())
    frame_to_i = {f: i for i, f in enumerate(frames)}
    resid_to_j = {r: j for j, r in enumerate(model_resids)}

    arr_deg = np.full((len(frames), len(model_resids), 2), np.nan, dtype=float)

    sub = internal_df[internal_df["resid"].isin(model_resids)]
    for _, row in sub.iterrows():
        i = frame_to_i[row["frame"]]
        j = resid_to_j[row["resid"]]
        arr_deg[i, j, 0] = row["phi"]
        arr_deg[i, j, 1] = row["psi"]

    if np.isnan(arr_deg).any():
        raise ValueError("Missing phi/psi values after extracting model residues.")

    if angle_unit == "degrees":
        return arr_deg, list(model_resids)
    if angle_unit == "radians":
        return np.deg2rad(arr_deg), list(model_resids)

    raise ValueError("angle_unit must be 'degrees' or 'radians'.")
