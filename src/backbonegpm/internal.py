import numpy as np
import pandas as pd
import MDAnalysis as mda


def bond_length(p1, p2):
    return np.linalg.norm(p2 - p1)


def bond_angle_deg(p1, p2, p3):
    """
    Angle p1-p2-p3 in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def dihedral_deg(p0, p1, p2, p3):
    """
    Standard signed dihedral angle in degrees.
    Matches the usual convention used by VMD / MDAnalysis.
    """
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0:
        return np.nan
    b1 = b1 / b1_norm

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv == 0 or nw == 0:
        return np.nan

    v = v / nv
    w = w / nw

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def get_backbone_atoms_by_residue(universe, selection="protein"):
    """
    Return a list of dictionaries, one per residue, containing:
        resindex, resid, resname, N, CA, C
    Skips residues missing any backbone atom.
    """
    ag = universe.select_atoms(selection)
    residues = ag.residues

    out = []
    for res in residues:
        try:
            N = res.atoms.select_atoms("name N")
            CA = res.atoms.select_atoms("name CA")
            C = res.atoms.select_atoms("name C")
            if len(N) != 1 and len(CA) != 1 and len(C) != 1:
                continue
            out.append(
                {
                    "resindex": res.ix,
                    "resid": res.resid,
                    "resname": res.resname,
                    "N": N[0],
                    "CA": CA[0],
                    "C": C[0],
                }
            )
        except Exception:
            continue
    return out


def compute_backbone_internal_coordinates(universe, selection="protein", frame=None):
    """
    Compute per-residue backbone bond lengths, angles, and dihedrals.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    selection : str
        Typically "protein" or another selection containing the residues of interest.
    frame : int or None
        If provided, move to that trajectory frame first.

    Returns
    -------
    df : pandas.DataFrame
        One row per residue with columns for available internal coordinates.
    """
    if frame is not None:
        universe.trajectory[frame]

    residues = get_backbone_atoms_by_residue(universe, selection=selection)
    n = len(residues)

    rows = []

    for i in range(n):
        row = {
            "i": i,
            "resid": residues[i]["resid"],
            "resname": residues[i]["resname"],
            "CN": np.nan,          # C_{i-1} - N_i
            "NCA": np.nan,         # N_i - CA_i
            "CAC": np.nan,         # CA_i - C_i
            "C_N_CA": np.nan,      # C_{i-1} - N_i - CA_i
            "N_CA_C": np.nan,      # N_i - CA_i - C_i
            "CA_C_N": np.nan,      # CA_i - C_i - N_{i+1}
            "phi": np.nan,         # C_{i-1} - N_i - CA_i - C_i
            "psi": np.nan,         # N_i - CA_i - C_i - N_{i+1}
            "omega": np.nan,       # CA_i - C_i - N_{i+1} - CA_{i+1}
        }

        N_i = residues[i]["N"].position
        CA_i = residues[i]["CA"].position
        C_i = residues[i]["C"].position

        # Intra-residue bond lengths
        row["NCA"] = bond_length(N_i, CA_i)
        row["CAC"] = bond_length(CA_i, C_i)

        # Intra-residue bond angle
        row["N_CA_C"] = bond_angle_deg(N_i, CA_i, C_i)

        # Previous-residue quantities
        if i > 0:
            C_prev = residues[i - 1]["C"].position

            row["CN"] = bond_length(C_prev, N_i)
            row["C_N_CA"] = bond_angle_deg(C_prev, N_i, CA_i)
            row["phi"] = dihedral_deg(C_prev, N_i, CA_i, C_i)

        # Next-residue quantities
        if i < n - 1:
            N_next = residues[i + 1]["N"].position
            CA_next = residues[i + 1]["CA"].position

            row["CA_C_N"] = bond_angle_deg(CA_i, C_i, N_next)
            row["psi"] = dihedral_deg(N_i, CA_i, C_i, N_next)
            row["omega"] = dihedral_deg(CA_i, C_i, N_next, CA_next)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_backbone_internal_coordinates(df):
    """
    Return summary statistics for the numeric internal-coordinate columns.
    """
    numeric_cols = [
        "CN", "NCA", "CAC",
        "C_N_CA", "N_CA_C", "CA_C_N",
        "phi", "psi", "omega"
    ]
    return df[numeric_cols].agg(["mean", "std", "min", "max"]).T

def get_single_atom(residue, atom_name):
    """
    Return a single atom from a residue by name, or None if absent/non-unique.
    """
    atoms = residue.atoms.select_atoms(f"name {atom_name}")
    if len(atoms) != 1:
        return None
    return atoms[0]

def get_capped_backbone_residues(universe, selection="all"):
    """
    Build a residue list suitable for capped peptide backbone analysis.

    Keeps:
      - ACE: only atom C
      - standard residues: N, CA, C
      - NME: only atom N

    Returns
    -------
    residues : list of dict
        Each dict has keys:
          resid, resname, N, CA, C
        with missing atoms set to None.
    """
    ag = universe.select_atoms(selection)
    residues = ag.residues

    out = []
    for res in residues:
        resname = res.resname.strip().upper()

        entry = {
            "resid": res.resid,
            "resname": res.resname,
            "N": None,
            "CA": None,
            "C": None,
        }

        if resname == "ACE":
            entry["C"] = get_single_atom(res, "C")
            if entry["C"] is not None:
                out.append(entry)

        elif resname == "NME":
            entry["N"] = get_single_atom(res, "N")
            if entry["N"] is not None:
                out.append(entry)

        else:
            entry["N"] = get_single_atom(res, "N")
            entry["CA"] = get_single_atom(res, "CA")
            entry["C"] = get_single_atom(res, "C")

            # standard amino acid backbone residue must have all 3
            if entry["N"] is not None and entry["CA"] is not None and entry["C"] is not None:
                out.append(entry)

    return out


def compute_capped_backbone_internal_coordinates(universe, selection="all", frame=None):
    """
    Compute backbone internal coordinates for a capped peptide:
        ACE - ALA - ALA - ... - NME

    For dialanine tripeptide (ACE-ALA-ALA-NME), this gives:
      - 3 CN-type bonds:
            ACE C - ALA1 N
            ALA1 C - ALA2 N
            ALA2 C - NME N
      - 2 NCA bonds
      - 2 CAC bonds
      - 2 phi
      - 2 psi
      - 1 omega
      - associated angles

    Returns
    -------
    df : pandas.DataFrame
        One row per residue-like unit (ACE, amino acid residues, NME).
    """
    if frame is not None:
        universe.trajectory[frame]

    residues = get_capped_backbone_residues(universe, selection=selection)
    n = len(residues)

    rows = []
    for i in range(n):
        res = residues[i]

        row = {
            "i": i,
            "resid": res["resid"],
            "resname": res["resname"],

            # bonds
            "CN": np.nan,      # C_{i-1} - N_i
            "NCA": np.nan,     # N_i - CA_i
            "CAC": np.nan,     # CA_i - C_i

            # angles
            "C_N_CA": np.nan,  # C_{i-1} - N_i - CA_i
            "N_CA_C": np.nan,  # N_i - CA_i - C_i
            "CA_C_N": np.nan,  # CA_i - C_i - N_{i+1}

            # dihedrals
            "phi": np.nan,     # C_{i-1} - N_i - CA_i - C_i
            "psi": np.nan,     # N_i - CA_i - C_i - N_{i+1}
            "omega": np.nan,   # CA_i - C_i - N_{i+1} - CA_{i+1}
        }

        N_i = res["N"].position if res["N"] is not None else None
        CA_i = res["CA"].position if res["CA"] is not None else None
        C_i = res["C"].position if res["C"] is not None else None

        # intra-residue quantities (only for standard residues)
        if N_i is not None and CA_i is not None:
            row["NCA"] = bond_length(N_i, CA_i)

        if CA_i is not None and C_i is not None:
            row["CAC"] = bond_length(CA_i, C_i)

        if N_i is not None and CA_i is not None and C_i is not None:
            row["N_CA_C"] = bond_angle_deg(N_i, CA_i, C_i)

        # previous-residue quantities
        if i > 0:
            prev = residues[i - 1]
            C_prev = prev["C"].position if prev["C"] is not None else None

            if C_prev is not None and N_i is not None:
                row["CN"] = bond_length(C_prev, N_i)

            if C_prev is not None and N_i is not None and CA_i is not None:
                row["C_N_CA"] = bond_angle_deg(C_prev, N_i, CA_i)
                if C_i is not None:
                    row["phi"] = dihedral_deg(C_prev, N_i, CA_i, C_i)

        # next-residue quantities
        if i < n - 1:
            nxt = residues[i + 1]
            N_next = nxt["N"].position if nxt["N"] is not None else None
            CA_next = nxt["CA"].position if nxt["CA"] is not None else None

            if CA_i is not None and C_i is not None and N_next is not None:
                row["CA_C_N"] = bond_angle_deg(CA_i, C_i, N_next)

            if N_i is not None and CA_i is not None and C_i is not None and N_next is not None:
                row["psi"] = dihedral_deg(N_i, CA_i, C_i, N_next)

            if CA_i is not None and C_i is not None and N_next is not None and CA_next is not None:
                row["omega"] = dihedral_deg(CA_i, C_i, N_next, CA_next)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_backbone_over_trajectory(
    topology_or_universe,
    trajectory=None,
    selection="protein",
    cluster_ids=None,
    capped=True,
):
    if hasattr(topology_or_universe, "trajectory"):
        u = topology_or_universe
    else:
        u = mda.Universe(topology_or_universe, trajectory) if trajectory else mda.Universe(topology_or_universe)

    all_frames = []
    for ts in u.trajectory:
        if capped:
            df = compute_capped_backbone_internal_coordinates(u, selection=selection)
        else:
            df = compute_backbone_internal_coordinates(u, selection=selection)

        df["frame"] = ts.frame
        if cluster_ids is not None:
            df["macrostate_id"] = cluster_ids[ts.frame]
        all_frames.append(df)

    return pd.concat(all_frames, ignore_index=True)
