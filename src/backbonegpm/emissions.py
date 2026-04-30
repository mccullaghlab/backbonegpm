import numpy as np
import pandas as pd

NORMAL_COLS = ("CN", "NCA", "CAC", "C_N_CA", "N_CA_C", "CA_C_N")
# ----------------------------------------------------------------------
# Local-geometry emissions
# ----------------------------------------------------------------------
def fit_von_mises_deg(angle_deg):
    x = np.deg2rad(pd.Series(angle_deg).dropna().to_numpy())
    if len(x) == 0:
        return {"mu_deg": np.nan, "kappa": np.nan, "Rbar": np.nan, "n": 0}

    C = np.mean(np.cos(x))
    S = np.mean(np.sin(x))
    mu = np.arctan2(S, C)
    Rbar = np.sqrt(C**2 + S**2)

    if Rbar < 1e-8:
        kappa = 0.0
    elif Rbar < 0.53:
        kappa = 2 * Rbar + Rbar**3 + 5 * Rbar**5 / 6
    elif Rbar < 0.85:
        kappa = -0.4 + 1.39 * Rbar + 0.43 / (1 - Rbar)
    else:
        kappa = 1 / (Rbar**3 - 4 * Rbar**2 + 3 * Rbar)

    return {"mu_deg": np.rad2deg(mu), "kappa": kappa, "Rbar": Rbar, "n": len(x)}


def fit_local_geometry_emissions(internal_df, z, model_resids, normal_cols=NORMAL_COLS):
    df = internal_df.copy()
    z = np.asarray(z, dtype=int)
    resid_to_col = {resid: j for j, resid in enumerate(model_resids)}

    df["microstate"] = pd.NA
    for resid, col in resid_to_col.items():
        mask = df["resid"].eq(resid)
        frames = df.loc[mask, "frame"].to_numpy(dtype=int)
        # This assumes z rows are in the same order as the filtered macrostate frames.
        # In production, pass macro-local frame_to_row mapping explicitly.
        unique_frames = np.array(sorted(df.loc[df["resid"].isin(model_resids), "frame"].unique()))
        frame_to_local = {f: i for i, f in enumerate(unique_frames)}
        local_rows = np.array([frame_to_local[f] for f in frames], dtype=int)
        df.loc[mask, "microstate"] = z[local_rows, col]

    df["microstate"] = df["microstate"].astype("Int64")

    normal_params = {}
    for resid in model_resids:
        normal_params[resid] = {}
        df_res = df[df["resid"] == resid]
        for k, g in df_res.groupby("microstate"):
            k = int(k)
            normal_params[resid][k] = {}
            for col in normal_cols:
                vals = g[col].dropna().to_numpy()
                normal_params[resid][k][col] = {
                    "mu": np.mean(vals) if len(vals) else np.nan,
                    "sigma": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                    "n": len(vals),
                }

    omega_params = {}
    if len(model_resids) >= 2:
        for j, resid_left in enumerate(model_resids[:-1]):
            df_omega = df[df["resid"] == resid_left].copy()
            df_omega = df_omega[df_omega["omega"].notna()]
            if len(df_omega) == 0:
                continue
            frames = df_omega["frame"].to_numpy(dtype=int)
            unique_frames = np.array(sorted(df.loc[df["resid"].isin(model_resids), "frame"].unique()))
            frame_to_local = {f: i for i, f in enumerate(unique_frames)}
            local_rows = np.array([frame_to_local[f] for f in frames], dtype=int)
            df_omega["z_left"] = z[local_rows, j]
            df_omega["z_right"] = z[local_rows, j + 1]
            for (k, l), g in df_omega.groupby(["z_left", "z_right"]):
                omega_params[(int(k), int(l), int(j))] = fit_von_mises_deg(g["omega"])

    return normal_params, omega_params


def sample_normal(mu, sigma, size, rng, min_sigma=1e-6):
    if np.isnan(mu):
        return np.full(size, np.nan)
    if np.isnan(sigma) or sigma < min_sigma:
        return np.full(size, mu)
    return rng.normal(mu, sigma, size=size)


def sample_von_mises_deg(mu_deg, kappa, size, rng):
    if np.isnan(mu_deg):
        return np.full(size, np.nan)
    if np.isnan(kappa) or kappa < 1e-8:
        return rng.uniform(-180.0, 180.0, size=size)
    return np.rad2deg(rng.vonmises(np.deg2rad(mu_deg), kappa, size=size))


def sample_local_geometry_from_emissions(
    z_gen,
    model_resids,
    emission_params,
    normal_cols=NORMAL_COLS,
    random_state=None,
    use_means=False,
):
    rng = np.random.default_rng(random_state)
    z_gen = np.asarray(z_gen, dtype=int)
    n_frames, n_res = z_gen.shape
    normal_params = emission_params["normal_params"]
    omega_params = emission_params.get("omega_params", {})

    rows = []
    for j, resid in enumerate(model_resids):
        states_j = z_gen[:, j]
        data = {"frame": np.arange(n_frames), "resid": resid, "microstate": states_j}
        for col in normal_cols:
            vals = np.empty(n_frames)
            for k in np.unique(states_j):
                mask = states_j == k
                p = normal_params[resid][int(k)][col]
                vals[mask] = p["mu"] if use_means else sample_normal(p["mu"], p["sigma"], mask.sum(), rng)
            data[col] = vals
        rows.append(pd.DataFrame(data))

    local_df = pd.concat(rows, ignore_index=True)
    local_df["omega"] = np.nan

    if n_res >= 2 and omega_params:
        for j in range(n_res - 1):
            resid_left = model_resids[j]
            z_left = z_gen[:, j]
            z_right = z_gen[:, j + 1]
            omega_vals = np.empty(n_frames)
            omega_vals[:] = np.nan
            for k in np.unique(z_left):
                for l in np.unique(z_right):
                    mask = (z_left == k) & (z_right == l)
                    if not np.any(mask):
                        continue
                    p = omega_params.get((int(k), int(l), int(j))) or omega_params.get((int(k), int(l)))
                    if p is not None:
                        omega_vals[mask] = p["mu_deg"] if use_means else sample_von_mises_deg(
                            p["mu_deg"], p["kappa"], mask.sum(), rng
                        )
            local_df.loc[local_df["resid"].eq(resid_left), "omega"] = omega_vals

    return local_df


def prepare_builder_inputs_from_sampled_local_geometry(local_df, phi_psi_deg, model_resids):
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

    omega = np.full((n_frames, max(n_res - 1, 0)), np.nan)
    for j, resid in enumerate(model_resids[:-1]):
        g = local_df[local_df["resid"].eq(resid)].sort_values("frame")
        omega[:, j] = g["omega"].to_numpy()
    params["omega"] = omega
    return params

