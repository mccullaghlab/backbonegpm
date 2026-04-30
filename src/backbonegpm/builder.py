import numpy as np
import pandas as pd


def prepare_builder_inputs_from_sampled_local_geometry(
    local_df,
    phi_psi_deg,
    model_resids,
):
    n_frames, n_res, _ = phi_psi_deg.shape

    colmap = {
        "CN": "d_CN",
        "NCA": "d_NCA",
        "CAC": "d_CAC",
        "C_N_CA": "ang_C_N_CA",
        "N_CA_C": "ang_N_CA_C",
        "CA_C_N": "ang_CA_C_N",
    }

    params = {}

    for df_col, builder_key in colmap.items():
        arr = np.full((n_frames, n_res), np.nan)

        for j, resid in enumerate(model_resids):
            g = local_df[local_df["resid"].eq(resid)].sort_values("frame")
            arr[:, j] = g[df_col].to_numpy()

        params[builder_key] = arr

    # omega should usually be one per peptide bond: shape (n_frames, n_res - 1)
    omega = np.full((n_frames, n_res - 1), np.nan)

    for j, resid in enumerate(model_resids[:-1]):
        g = local_df[local_df["resid"].eq(resid)].sort_values("frame")
        omega[:, j] = g["omega"].to_numpy()

    params["omega"] = omega

    return params

def _slice_param_for_sample(x, s, n_samples):
    """
    If x is sample-dependent with leading dimension n_samples, return x[s].
    Otherwise return x unchanged.
    """
    if x is None:
        return None

    arr = np.asarray(x)

    if arr.ndim >= 1 and arr.shape[0] == n_samples:
        return arr[s]

    return x

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Encountered near-zero vector in _unit().")
    return v / n


def _as_1d_parameter_array(x, length, name):
    """
    Convert scalar or array-like parameter into a 1D float array of given length.
    """
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        return np.full(length, float(arr), dtype=float)

    arr = np.ravel(arr)
    if len(arr) != length:
        raise ValueError(
            f"{name} must be a scalar or have length {length}, got shape {np.shape(x)}"
        )
    return arr.astype(float)


def place_atom(a, b, c, bond_length, bond_angle_deg, dihedral_deg):
    """
    Place atom D from internal coordinates relative to atoms A-B-C.

    Definitions
    ----------
    bond length  = |C-D|
    bond angle   = angle(B, C, D)
    dihedral     = dihedral(A, B, C, D)

    Angles are in degrees.

    This version is shifted by 180 degrees so that the resulting dihedrals
    match standard VMD / MDAnalysis conventions.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    theta = np.deg2rad(bond_angle_deg)
    tau = np.deg2rad(dihedral_deg + 180.0)

    bc = _unit(c - b)
    n = _unit(np.cross(bc, b - a))
    m = np.cross(n, bc)

    d = (
        c
        + (-bond_length * np.cos(theta)) * bc
        + (bond_length * np.sin(theta)) * (np.cos(tau) * m + np.sin(tau) * n)
    )
    return d


def build_backbone_with_local_geometry(
    phi_psi,
    omega=180.0,
    d_CN=1.329,
    d_NCA=1.458,
    d_CAC=1.525,
    ang_C_N_CA=121.7,
    ang_N_CA_C=110.4,
    ang_CA_C_N=116.2,
    d_CN_terminal=None,
    ang_CA_C_N_terminal=None,
    angle_unit="degrees",
):
    """
    Build backbone coordinates with an extra leading C atom and trailing N atom.

    Geometry layout
    ---------------
    Output contains:
        C_prev, [N_i, CA_i, C_i for i=1..n_res], N_next

    so the total number of atoms is:
        3*n_res + 2

    Input torsions
    --------------
    phi_psi[..., 0] = phi_i  = dihedral(C_{i-1}, N_i, CA_i, C_i)
    phi_psi[..., 1] = psi_i  = dihedral(N_i, CA_i, C_i, N_{i+1})

    Parameters
    ----------
    phi_psi : ndarray
        Shape (n_res, 2) or (n_samples, n_res, 2)

    omega : scalar or array-like
        Peptide-bond torsion(s), one per peptide bond.
        Length should be n_res-1 if array-like.

    d_CN : scalar or array-like
        C_{i-1} -- N_i bond lengths, one per residue. Length n_res.

    d_NCA : scalar or array-like
        N_i -- CA_i bond lengths, one per residue. Length n_res.

    d_CAC : scalar or array-like
        CA_i -- C_i bond lengths, one per residue. Length n_res.

    ang_C_N_CA : scalar or array-like
        Angle C_{i-1} - N_i - CA_i, one per residue. Length n_res.

    ang_N_CA_C : scalar or array-like
        Angle N_i - CA_i - C_i, one per residue. Length n_res.

    ang_CA_C_N : scalar or array-like
        Angle CA_i - C_i - N_{i+1}, one per residue. Length n_res.

    d_CN_terminal : scalar or None
        Optional terminal bond length for C_n -- N_cap. If None, uses d_CN[-1].

    ang_CA_C_N_terminal : scalar or None
        Optional terminal bond angle for CA_n - C_n - N_cap. If None, uses ang_CA_C_N[-1].

    angle_unit : {"degrees", "radians"}
        Units for phi_psi, omega, and angles. Bond lengths stay in Å.

    Returns
    -------
    coords : dict
        For single input:
            C_prev : (1, 3)
            N      : (n_res, 3)
            CA     : (n_res, 3)
            C      : (n_res, 3)
            N_next : (1, 3)
            all    : (3*n_res + 2, 3)

        For batched input:
            C_prev : (n_samples, 1, 3)
            N      : (n_samples, n_res, 3)
            CA     : (n_samples, n_res, 3)
            C      : (n_samples, n_res, 3)
            N_next : (n_samples, 1, 3)
            all    : (n_samples, 3*n_res + 2, 3)
    """
    phi_psi = np.asarray(phi_psi, dtype=float)

    if phi_psi.ndim == 2:
        if phi_psi.shape[1] != 2:
            raise ValueError(f"Expected shape (n_res, 2), got {phi_psi.shape}")
        return _build_backbone_single_with_local_geometry(
            phi_psi=phi_psi,
            omega=omega,
            d_CN=d_CN,
            d_NCA=d_NCA,
            d_CAC=d_CAC,
            ang_C_N_CA=ang_C_N_CA,
            ang_N_CA_C=ang_N_CA_C,
            ang_CA_C_N=ang_CA_C_N,
            d_CN_terminal=d_CN_terminal,
            ang_CA_C_N_terminal=ang_CA_C_N_terminal,
            angle_unit=angle_unit,
        )

    elif phi_psi.ndim == 3:
        if phi_psi.shape[2] != 2:
            raise ValueError(f"Expected shape (n_samples, n_res, 2), got {phi_psi.shape}")

        n_samples, n_res, _ = phi_psi.shape

        C_prev_all = np.empty((n_samples, 1, 3), dtype=float)
        N_all = np.empty((n_samples, n_res, 3), dtype=float)
        CA_all = np.empty((n_samples, n_res, 3), dtype=float)
        C_all = np.empty((n_samples, n_res, 3), dtype=float)
        N_next_all = np.empty((n_samples, 1, 3), dtype=float)
        flat_all = np.empty((n_samples, 3 * n_res + 2, 3), dtype=float)

        for s in range(n_samples):
            out = _build_backbone_single_with_local_geometry(
                phi_psi=phi_psi[s],
                omega=_slice_param_for_sample(omega, s, n_samples),
                d_CN=_slice_param_for_sample(d_CN, s, n_samples),
                d_NCA=_slice_param_for_sample(d_NCA, s, n_samples),
                d_CAC=_slice_param_for_sample(d_CAC, s, n_samples),
                ang_C_N_CA=_slice_param_for_sample(ang_C_N_CA, s, n_samples),
                ang_N_CA_C=_slice_param_for_sample(ang_N_CA_C, s, n_samples),
                ang_CA_C_N=_slice_param_for_sample(ang_CA_C_N, s, n_samples),
                d_CN_terminal=_slice_param_for_sample(d_CN_terminal, s, n_samples),
                ang_CA_C_N_terminal=_slice_param_for_sample(ang_CA_C_N_terminal, s, n_samples),
                angle_unit=angle_unit,
            )
            C_prev_all[s] = out["C_prev"]
            N_all[s] = out["N"]
            CA_all[s] = out["CA"]
            C_all[s] = out["C"]
            N_next_all[s] = out["N_next"]
            flat_all[s] = out["all"]

        return {
            "C_prev": C_prev_all,
            "N": N_all,
            "CA": CA_all,
            "C": C_all,
            "N_next": N_next_all,
            "all": flat_all,
        }

    else:
        raise ValueError(
            f"phi_psi must have shape (n_res, 2) or (n_samples, n_res, 2), got {phi_psi.shape}"
        )


def _build_backbone_single_with_local_geometry(
    phi_psi,
    omega,
    d_CN,
    d_NCA,
    d_CAC,
    ang_C_N_CA,
    ang_N_CA_C,
    ang_CA_C_N,
    d_CN_terminal=None,
    ang_CA_C_N_terminal=None,
    angle_unit="degrees",
):
    """
    Build one backbone with residue-specific local geometry.
    """
    n_res = phi_psi.shape[0]
    if n_res < 1:
        raise ValueError("Need at least one residue.")

    phi = phi_psi[:, 0].astype(float).copy()
    psi = phi_psi[:, 1].astype(float).copy()

    d_CN = _as_1d_parameter_array(d_CN, n_res, "d_CN")
    d_NCA = _as_1d_parameter_array(d_NCA, n_res, "d_NCA")
    d_CAC = _as_1d_parameter_array(d_CAC, n_res, "d_CAC")

    ang_C_N_CA = _as_1d_parameter_array(ang_C_N_CA, n_res, "ang_C_N_CA")
    ang_N_CA_C = _as_1d_parameter_array(ang_N_CA_C, n_res, "ang_N_CA_C")
    ang_CA_C_N = _as_1d_parameter_array(ang_CA_C_N, n_res, "ang_CA_C_N")

    if n_res > 1:
        omega = _as_1d_parameter_array(omega, n_res - 1, "omega")
    else:
        omega = np.empty(0, dtype=float)

    if angle_unit == "radians":
        phi = np.degrees(phi)
        psi = np.degrees(psi)
        omega = np.degrees(omega)
        ang_C_N_CA = np.degrees(ang_C_N_CA)
        ang_N_CA_C = np.degrees(ang_N_CA_C)
        ang_CA_C_N = np.degrees(ang_CA_C_N)
        if d_CN_terminal is not None:
            d_CN_terminal = float(d_CN_terminal)
        if ang_CA_C_N_terminal is not None:
            ang_CA_C_N_terminal = np.degrees(ang_CA_C_N_terminal)
    elif angle_unit != "degrees":
        raise ValueError("angle_unit must be 'degrees' or 'radians'.")

    if d_CN_terminal is None:
        d_CN_terminal = d_CN[-1]
    else:
        d_CN_terminal = float(d_CN_terminal)

    if ang_CA_C_N_terminal is None:
        ang_CA_C_N_terminal = ang_CA_C_N[-1]
    else:
        ang_CA_C_N_terminal = float(ang_CA_C_N_terminal)

    C_prev = np.zeros((1, 3), dtype=float)
    N = np.zeros((n_res, 3), dtype=float)
    CA = np.zeros((n_res, 3), dtype=float)
    C = np.zeros((n_res, 3), dtype=float)
    N_next = np.zeros((1, 3), dtype=float)

    # Seed: C0, N1, CA1
    C_prev[0] = np.array([0.0, 0.0, 0.0])
    N[0] = np.array([d_CN[0], 0.0, 0.0])

    alpha = np.deg2rad(180.0 - ang_C_N_CA[0])
    CA[0] = N[0] + np.array([
        d_NCA[0] * np.cos(alpha),
        d_NCA[0] * np.sin(alpha),
        0.0
    ])

    # Place C1 using phi1
    C[0] = place_atom(
        C_prev[0], N[0], CA[0],
        bond_length=d_CAC[0],
        bond_angle_deg=ang_N_CA_C[0],
        dihedral_deg=phi[0]
    )

    # Build residues 2..n_res
    for i in range(n_res - 1):
        # psi_i places N_{i+1}
        N[i + 1] = place_atom(
            N[i], CA[i], C[i],
            bond_length=d_CN[i + 1],
            bond_angle_deg=ang_CA_C_N[i],
            dihedral_deg=psi[i]
        )

        # omega_i places CA_{i+1}
        CA[i + 1] = place_atom(
            CA[i], C[i], N[i + 1],
            bond_length=d_NCA[i + 1],
            bond_angle_deg=ang_C_N_CA[i + 1],
            dihedral_deg=omega[i]
        )

        # phi_{i+1} places C_{i+1}
        C[i + 1] = place_atom(
            C[i], N[i + 1], CA[i + 1],
            bond_length=d_CAC[i + 1],
            bond_angle_deg=ang_N_CA_C[i + 1],
            dihedral_deg=phi[i + 1]
        )

    # Final psi_n places N_cap
    N_next[0] = place_atom(
        N[-1], CA[-1], C[-1],
        bond_length=d_CN_terminal,
        bond_angle_deg=ang_CA_C_N_terminal,
        dihedral_deg=psi[-1]
    )

    flat = np.empty((3 * n_res + 2, 3), dtype=float)
    flat[0] = C_prev[0]
    for i in range(n_res):
        flat[1 + 3 * i] = N[i]
        flat[2 + 3 * i] = CA[i]
        flat[3 + 3 * i] = C[i]
    flat[-1] = N_next[0]

    return {
        "C_prev": C_prev,
        "N": N,
        "CA": CA,
        "C": C,
        "N_next": N_next,
        "all": flat,
    }


def prepare_builder_inputs_from_residue_means_caps(residue_means, phi_psi_samples):
    """
    Convert residue_means into parameter arrays for the capped-backbone builder.

    Expected residue_means rows for a capped peptide:
        ACE, standard residue(s), NME

    The true builder residues are the rows with NCA and CAC defined.

    Parameters
    ----------
    residue_means : pandas.DataFrame
        Expected columns:
            resid, resname,
            CN, NCA, CAC,
            C_N_CA, N_CA_C, CA_C_N,
            phi, psi, omega

    phi_psi_samples : ndarray
        Shape (n_res, 2) or (n_samples, n_res, 2)

    Returns
    -------
    params : dict
        Keyword args for build_backbone_with_local_geometry(...)
    """
    df = residue_means.copy().reset_index(drop=True)

    # "True" residues are the ones with N, CA, C
    std = df[df["NCA"].notna() & df["CAC"].notna()].copy().reset_index(drop=True)
    n_res = len(std)

    if n_res < 1:
        raise ValueError("No standard residues found in residue_means.")

    phi_psi_samples = np.asarray(phi_psi_samples, dtype=float)

    if phi_psi_samples.ndim == 2:
        expected = (n_res, 2)
        if phi_psi_samples.shape != expected:
            raise ValueError(
                f"Expected phi_psi_samples shape {expected}, got {phi_psi_samples.shape}"
            )
    elif phi_psi_samples.ndim == 3:
        expected_tail = (n_res, 2)
        if phi_psi_samples.shape[1:] != expected_tail:
            raise ValueError(
                f"Expected phi_psi_samples trailing shape {expected_tail}, "
                f"got {phi_psi_samples.shape[1:]}"
            )
    else:
        raise ValueError(
            "phi_psi_samples must have shape (n_res, 2) or (n_samples, n_res, 2)"
        )

    params = {
        "d_CN": std["CN"].to_numpy(dtype=float),
        "d_NCA": std["NCA"].to_numpy(dtype=float),
        "d_CAC": std["CAC"].to_numpy(dtype=float),
        "ang_C_N_CA": std["C_N_CA"].to_numpy(dtype=float),
        "ang_N_CA_C": std["N_CA_C"].to_numpy(dtype=float),
        "ang_CA_C_N": std["CA_C_N"].to_numpy(dtype=float),
    }

    if n_res > 1:
        omega = std["omega"].to_numpy(dtype=float)[:-1]
        if np.any(np.isnan(omega)):
            raise ValueError("Missing omega values for internal peptide bond(s).")
        params["omega"] = omega
    else:
        params["omega"] = np.empty(0, dtype=float)

    # Optional terminal cap row, e.g. NME row with CN defined
    cap_rows = df[df["NCA"].isna() & df["CAC"].isna() & df["CN"].notna()]
    if len(cap_rows) > 0:
        terminal_row = cap_rows.iloc[-1]
        params["d_CN_terminal"] = float(terminal_row["CN"])

        # Usually terminal CA_C_N is not separately available for NME row,
        # so only use it if present.
        if "CA_C_N" in terminal_row.index and pd.notna(terminal_row["CA_C_N"]):
            params["ang_CA_C_N_terminal"] = float(terminal_row["CA_C_N"])

    return params


def internal_coords_from_reduced_backbone(coords):
    """
    Recompute internal coordinates from reduced backbone coordinates.

    Input coords should be the output dict from build_backbone_with_local_geometry.

    Returns
    -------
    df : pandas.DataFrame
        One row per true residue.
    """
    xyz = np.asarray(coords["all"], dtype=float)

    if xyz.ndim != 2:
        raise ValueError("This checker expects a single structure, not a batch.")

    if (xyz.shape[0] - 2) % 3 != 0:
        raise ValueError("Expected reduced capped-backbone layout: 3*n_res + 2 atoms.")

    n_res = (xyz.shape[0] - 2) // 3

    def bond_length(p1, p2):
        return np.linalg.norm(p2 - p1)

    def bond_angle_deg(p1, p2, p3):
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

    C_prev = xyz[0]
    N_next = xyz[-1]

    N = np.zeros((n_res, 3), dtype=float)
    CA = np.zeros((n_res, 3), dtype=float)
    C = np.zeros((n_res, 3), dtype=float)

    for i in range(n_res):
        base = 1 + 3 * i
        N[i] = xyz[base]
        CA[i] = xyz[base + 1]
        C[i] = xyz[base + 2]

    rows = []
    for i in range(n_res):
        row = {
            "resid_like": i + 1,
            "CN": np.nan,
            "NCA": bond_length(N[i], CA[i]),
            "CAC": bond_length(CA[i], C[i]),
            "C_N_CA": np.nan,
            "N_CA_C": bond_angle_deg(N[i], CA[i], C[i]),
            "CA_C_N": np.nan,
            "phi": np.nan,
            "psi": np.nan,
            "omega": np.nan,
            "CN_terminal": np.nan,
        }

        C_prev_i = C_prev if i == 0 else C[i - 1]
        row["CN"] = bond_length(C_prev_i, N[i])
        row["C_N_CA"] = bond_angle_deg(C_prev_i, N[i], CA[i])
        row["phi"] = dihedral_deg(C_prev_i, N[i], CA[i], C[i])

        if i < n_res - 1:
            row["CA_C_N"] = bond_angle_deg(CA[i], C[i], N[i + 1])
            row["psi"] = dihedral_deg(N[i], CA[i], C[i], N[i + 1])
            row["omega"] = dihedral_deg(CA[i], C[i], N[i + 1], CA[i + 1])
        else:
            row["CA_C_N"] = bond_angle_deg(CA[i], C[i], N_next)
            row["psi"] = dihedral_deg(N[i], CA[i], C[i], N_next)
            row["CN_terminal"] = bond_length(C[i], N_next)

        rows.append(row)

    return pd.DataFrame(rows)

