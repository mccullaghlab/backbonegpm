"""
Package for a hierarchical shapeGMM -> BVVMMM -> GPM model.

"""
from __future__ import annotations

from .builder import build_backbone_with_local_geometry, prepare_builder_inputs_from_sampled_local_geometry
from .emissions import fit_local_geometry_emissions, sample_local_geometry_from_emissions
from .features import load_universe, trajectory_atom_positions
from .internal import compute_backbone_over_trajectory
from .selection import extract_phi_psi_from_internal_df, infer_model_resids

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


NORMAL_COLS = ("CN", "NCA", "CAC", "C_N_CA", "N_CA_C", "CA_C_N")


@dataclass
class FitConfig:
    n_macrostates: int
    microstate_components: Sequence[np.ndarray] | dict[int, Sequence[np.ndarray]]
    macro_selection: str = "name N CA C"
    internal_selection: str = "all"
    delta_fit: int = 1
    dtype: Any = None
    device: Any = None
    bvvmmm_n_attempts: int = 10
    random_state: Optional[int] = None
    fit_local_geometry: bool = True
    capped: bool = True
    verbose: bool = False
    verbose_submodels: bool = False


@dataclass
class MacrostateModel:
    bvvmmm: Any
    gpm: Any
    components: np.ndarray | Sequence[np.ndarray]
    normal_params: dict = field(default_factory=dict)
    omega_params: dict = field(default_factory=dict)
    model_resids: list[int] = field(default_factory=list)


class HierarchicalBackboneGPM:
    """
    High-level estimator for:

        trajectory -> shapeGMM macrostates
                   -> per-macrostate BVVMMM microstates
                   -> per-macrostate GPM coupling model
                   -> generated phi/psi and reduced capped backbones

    This class intentionally delegates heavy work to small helper functions:
      - feature extraction from MDAnalysis
      - internal-coordinate computation
      - local-geometry emission fitting
      - backbone coordinate construction
      - PDB writing
    """

    def __init__(self, config: FitConfig):
        self.config = config
        self.macro_model_: Any = None
        self.macrostate_ids_: Optional[np.ndarray] = None
        self.phi_psi_: Optional[np.ndarray] = None
        self.internal_df_: Optional[pd.DataFrame] = None
        self.model_resids_: Optional[list[int]] = None
        self.macrostates_: list[MacrostateModel] = []
        self.macrostate_probs_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        topology: str | Path,
        trajectory: str | Path | Sequence[str | Path] | None,
        config: FitConfig,
    ) -> "HierarchicalBackboneGPM":
        obj = cls(config)
        obj.topology_ = topology
        obj.trajectory_ = trajectory
        return obj

    # ------------------------------------------------------------------
    # Main fitting workflow
    # ------------------------------------------------------------------
    def fit(
        self,
        universe: Any | None = None,
        macro_features: np.ndarray | None = None,
        phi_psi: np.ndarray | None = None,
        internal_df: pd.DataFrame | None = None,
    ) -> "HierarchicalBackboneGPM":
        """
        Fit the full hierarchy.

        Parameters
        ----------
        universe
            Optional MDAnalysis Universe. If supplied, the class computes
            macrostate features, phi/psi, and local internal coordinates.
        macro_features
            Optional array passed directly to shapeGMM. Shape should be
            (n_frames, n_atoms, 3) 
        phi_psi
            Optional torsion array, shape (n_frames, n_model_residues, 2).
            Convention should match the BVVMMM implementation.
        internal_df
            Optional dataframe containing per-frame/per-residue internal
            coordinates: frame, resid, resname, CN, NCA, CAC, angles, phi, psi,
            omega.
        """
        cfg = self.config

        self._log("Starting hierarchical model fit.")

        if universe is None and hasattr(self, "topology_"):
            universe = self._load_universe(self.topology_, self.trajectory_)

        if macro_features is None:
            if universe is None:
                raise ValueError("Provide either universe or macro_features.")
            macro_features = self._compute_macro_features(universe, cfg.macro_selection)

        if phi_psi is None or internal_df is None:
            if universe is None:
                raise ValueError("Provide universe, or both phi_psi and internal_df.")
            self._log("Computing backbone internal coordinates from trajectory. This may take a while for large systems.")
            internal_df = self._compute_internal_df(universe)
            self._log("Extracting phi/psi values from internal coordinate dataframe.")
            phi_psi, model_resids = self._extract_phi_psi_from_internal_df(internal_df)
        else:
            model_resids = self._infer_model_resids(internal_df)

        self.phi_psi_ = np.asarray(phi_psi)
        self.internal_df_ = internal_df.copy()
        self.model_resids_ = list(model_resids)

        macro_features_fit = macro_features[:: cfg.delta_fit]
        self._log(f"Determining macrostates for {macro_features.shape[1]} atoms over {macro_features.shape[0]} frames using shapeGMM.")
        self._log(f"Fitting shapeGMM on {macro_features_fit.shape[0]} frames (delta_fit={cfg.delta_fit}).")
        self.macro_model_ = self._fit_shape_gmm(macro_features_fit)
        self.macrostate_ids_ = np.asarray(self.macro_model_.predict(macro_features), dtype=int)
        self._log("Assigned macrostates to all frames.")
        counts = np.bincount(self.macrostate_ids_, minlength=cfg.n_macrostates)
        self.macrostate_probs_ = counts / counts.sum()

        self.macrostates_ = []
        for m in range(cfg.n_macrostates):
            mask = self.macrostate_ids_ == m
            if not np.any(mask):
                raise ValueError(f"Macrostate {m} has zero assigned frames.")

            components_m = self._components_for_macrostate(m)
            self._log(f"Fitting macrostate {m + 1}/{cfg.n_macrostates}: {int(mask.sum())} frames.")
            bvvmmm = self._fit_bvvmmm(self.phi_psi_[mask], components_m)
            z_m = np.asarray(bvvmmm.predict_micro(self.phi_psi_[mask]), dtype=int)
            self._log(f"Fitting GPM for macrostate {m + 1}/{cfg.n_macrostates}.")
            gpm = self._fit_gpm(z_m, bvvmmm.components)

            mm = MacrostateModel(
                bvvmmm=bvvmmm,
                gpm=gpm,
                components=bvvmmm.components,
                model_resids=list(self.model_resids_),
            )

            if cfg.fit_local_geometry:
                self._log(f"Fitting local geometry emissions for macrostate {m + 1}/{cfg.n_macrostates}.")
                df_m = self.internal_df_[self.internal_df_["frame"].isin(np.where(mask)[0])].copy()
                mm.normal_params, mm.omega_params = fit_local_geometry_emissions(
                    internal_df=df_m,
                    z=z_m,
                    model_resids=self.model_resids_,
                )

            self.macrostates_.append(mm)

        self.is_fitted_ = True
        self._log("Hierarchical model fit completed.")
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def sample_microstates(
        self,
        n_samples: int | None = None,
        macrostate_ids: np.ndarray | None = None,
        enforce_training_macro_counts: bool = False,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        self._check_fitted()
        rng = np.random.default_rng(random_state)

        if macrostate_ids is None:
            if enforce_training_macro_counts:
                macrostate_ids = np.concatenate([
                    np.full(np.sum(self.macrostate_ids_ == m), m, dtype=int)
                    for m in range(len(self.macrostates_))
                ])
                rng.shuffle(macrostate_ids)
            else:
                if n_samples is None:
                    n_samples = len(self.macrostate_ids_)
                macrostate_ids = rng.choice(
                    len(self.macrostates_), size=n_samples, p=self.macrostate_probs_
                )
        else:
            macrostate_ids = np.asarray(macrostate_ids, dtype=int)

        z_by_macro = []
        for m, mm in enumerate(self.macrostates_):
            n_m = int(np.sum(macrostate_ids == m))
            z_by_macro.append(mm.gpm.sample(n_samples=n_m))

        return macrostate_ids, z_by_macro

    def generate(
        self,
        n_samples: int | None = None,
        build_backbone: bool = True,
        enforce_training_macro_counts: bool = False,
        independent: bool = False,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        self._check_fitted()
        macro_ids, z_by_macro = self.sample_microstates(
            n_samples=n_samples,
            enforce_training_macro_counts=enforce_training_macro_counts,
            random_state=random_state,
        )

        phi_psi_by_macro = []
        coords_by_macro = []
        local_geometry_by_macro = []

        for m, mm in enumerate(self.macrostates_):
            z_m = z_by_macro[m]
            if z_m.shape[0] == 0:
                phi_psi_by_macro.append(np.empty((0, len(self.model_resids_), 2)))
                continue

            if independent:
                phi_psi_m = mm.bvvmmm.generate_independent(z_m.shape[0])
            else:
                phi_psi_m = mm.bvvmmm.generate_from_micro_states(z_m)

            phi_psi_by_macro.append(phi_psi_m)

            if build_backbone:
                local_df = sample_local_geometry_from_emissions(
                    z_gen=z_m,
                    model_resids=mm.model_resids,
                    emission_params={
                        "normal_params": mm.normal_params,
                        "omega_params": mm.omega_params,
                    },
                    random_state=None if random_state is None else random_state + m,
                )
                local_geometry_by_macro.append(local_df)

                # BVVMMM samples appear to be radians in your snippets.
                phi_psi_deg = np.degrees(phi_psi_m)
                params = prepare_builder_inputs_from_sampled_local_geometry(
                    local_df=local_df,
                    phi_psi_deg=phi_psi_deg,
                    model_resids=mm.model_resids,
                )
                coords = build_backbone_with_local_geometry(phi_psi=phi_psi_deg, **params)
                coords_by_macro.append(coords)

        return {
            "macrostate_ids": macro_ids,
            "microstates_by_macrostate": z_by_macro,
            "phi_psi_by_macrostate": phi_psi_by_macro,
            "coords_by_macrostate": coords_by_macro,
            "local_geometry_by_macrostate": local_geometry_by_macro,
        }

    # ------------------------------------------------------------------
    # Dependency wrappers: keep imports local so package can have extras.
    # ------------------------------------------------------------------

    def _log(self, message: str):
        if self.config.verbose:
            print(f"[HierarchicalBackboneGPM] {message}")

    def _load_universe(self, topology, trajectory):
        return load_universe(topology, trajectory)

    def _fit_shape_gmm(self, X):
        import torch
        from shapeGMMTorch.utils import sgmm_fit_with_attempts

        dtype = self.config.dtype if self.config.dtype is not None else torch.float64
        device = self.config.device if self.config.device is not None else torch.device("cpu")
        return sgmm_fit_with_attempts(
            X,
            self.config.n_macrostates,
            dtype=dtype,
            device=device,
            verbose=self.config.verbose_submodels,
        )

    def _fit_bvvmmm(self, phi_psi_m, components_m):
        from multi import MultiIndSineBVvMMM

        model = MultiIndSineBVvMMM()
        model.fit(
            phi_psi_m,
            components=components_m,
            n_attempts=self.config.bvvmmm_n_attempts,
            plot=False,
            verbose=self.config.verbose_submodels,
        )
        model.refine(phi_psi_m)
        return model

    def _fit_gpm(self, z_m, components):
        from plgpm import PLGPM

        model = PLGPM(components)
        model.fit(z_m)
        return model

    def _compute_macro_features(self, universe, selection):
        return trajectory_atom_positions(universe, selection)

    def _compute_internal_df(self, universe):
        return compute_backbone_over_trajectory(
            universe,
            trajectory=None,
            selection=self.config.internal_selection,
            cluster_ids=None,
            capped=self.config.capped,
        )

    def _extract_phi_psi_from_internal_df(self, df):
        return extract_phi_psi_from_internal_df(df, angle_unit="radians")

    def _infer_model_resids(self, df):
        return infer_model_resids(df)

    def _components_for_macrostate(self, m):
        comps = self.config.microstate_components
        if isinstance(comps, dict):
            return comps[m]
        return comps[m]

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call .fit(...) first.")


