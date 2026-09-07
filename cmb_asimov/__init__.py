"""Produce smooth primary-CMB and CMB-lensing Asimov data vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """Generating calibration parameters matching the NERSC likelihood."""

    A_planck: float = 1.0
    calTE: float = 1.0
    calEE: float = 1.0
    P_act: float = 1.0
    Tcal: float = 1.0
    Ecal: float = 1.0


class CMBAsimovGenerator:
    """Map smooth CMB spectra to the NERSC primary and lensing data vectors."""

    def __init__(self, data_dir: str | Path | None = None):
        root = Path(data_dir) if data_dir is not None else Path(__file__).parent / "data"
        self.primary = np.load(root / "primary_cmb_assets.npz")
        self.lensing = np.load(root / "lensing_assets.npz")

    @staticmethod
    def _require_length(name: str, values: np.ndarray, length: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size < length:
            raise ValueError(
                f"{name} must be a one-dimensional array with at least {length} entries"
            )
        return array

    @staticmethod
    def _aberration(dl: np.ndarray, ell: np.ndarray, coefficient: float = -0.0004826) -> np.ndarray:
        cl = dl * (2.0 * np.pi) / (ell * (ell + 1.0))
        correction = -coefficient * ell * ell * (ell + 1.0) / (2.0 * np.pi) * np.gradient(cl)
        return dl + correction

    def primary_cmb(
        self,
        dl_tt: np.ndarray,
        dl_te: np.ndarray,
        dl_ee: np.ndarray,
        calibration: Calibration = Calibration(),
    ) -> dict[str, np.ndarray]:
        """Return CamSpec, ACT, and SPT smooth mean data vectors.

        Inputs are zero-based lensed D_ell arrays in microkelvin squared.
        """
        dl_tt = self._require_length("dl_tt", dl_tt, 8502)
        dl_te = self._require_length("dl_te", dl_te, 8502)
        dl_ee = self._require_length("dl_ee", dl_ee, 8502)

        a2 = calibration.A_planck**2
        camspec = np.concatenate(
            (
                dl_tt[self.primary["camspec_ell_tt"]] / a2,
                dl_te[self.primary["camspec_ell_te"]] / (a2 * calibration.calTE),
                dl_ee[self.primary["camspec_ell_ee"]] / (a2 * calibration.calEE),
            )
        )

        act = np.concatenate(
            (
                self.primary["act_window_tt"] @ dl_tt[self.primary["act_window_ell_tt"]] / a2,
                self.primary["act_window_te"]
                @ dl_te[self.primary["act_window_ell_te"]]
                / (a2 * calibration.P_act),
                self.primary["act_window_ee"]
                @ dl_ee[self.primary["act_window_ell_ee"]]
                / (a2 * calibration.P_act**2),
            )
        )

        ell = self.primary["spt_theory_ell"]
        tt_spt = self._aberration(dl_tt[ell], ell) / calibration.Tcal**2
        te_spt = self._aberration(dl_te[ell], ell) / (calibration.Tcal**2 * calibration.Ecal)
        ee_spt = self._aberration(dl_ee[ell], ell) / (calibration.Tcal**2 * calibration.Ecal**2)
        spt = np.concatenate(
            (
                self.primary["spt_window_tt"] @ tt_spt,
                self.primary["spt_window_te"] @ te_spt,
                self.primary["spt_window_ee"] @ ee_spt,
            )
        )
        return {"camspec": camspec, "act": act, "spt": spt}

    def lensing_cmb(
        self,
        cl_pp: np.ndarray,
        cl_tt: np.ndarray,
        cl_te: np.ndarray,
        cl_ee: np.ndarray,
        cl_bb: np.ndarray,
        *,
        act_calib: bool = False,
    ) -> np.ndarray:
        """Return the 35-element corrected ACT+Planck+SPT lensing mean vector.

        PP is raw dimensionless C_ell^{phi phi}. The lensed primary inputs are
        raw C_ell in microkelvin squared. All arrays are zero-based.
        """
        cl_pp = self._require_length("cl_pp", cl_pp, 3102)
        cl_tt = self._require_length("cl_tt", cl_tt, 3000)[:3000]
        cl_te = self._require_length("cl_te", cl_te, 3000)[:3000]
        cl_ee = self._require_length("cl_ee", cl_ee, 3000)[:3000]
        cl_bb = self._require_length("cl_bb", cl_bb, 3000)[:3000]

        ell = np.arange(cl_pp.size, dtype=np.float64)
        cl_kk = cl_pp * (ell * (ell + 1.0)) ** 2 / 4.0
        cl_kk_3000 = cl_kk[:3000]

        act_calibration = 1.0
        if act_calib:
            act_calibration = np.mean(cl_tt[1001:2000] / self.lensing["fid_cl_tt"][1001:2000])

        spectra = {"tt": cl_tt, "te": cl_te, "ee": cl_ee, "bb": cl_bb}
        blocks = []
        for experiment in ("act", "planck"):
            calibration = act_calibration if experiment == "act" else 1.0
            block = self.lensing[f"binmat_{experiment}_lensing"] @ cl_kk_3000
            block += self.lensing[f"kernel_{experiment}_kk"] @ (
                cl_kk_3000 - self.lensing["fid_cl_kk"]
            )
            for spectrum, values in spectra.items():
                block += self.lensing[f"kernel_{experiment}_{spectrum}"] @ (
                    values / calibration - self.lensing[f"fid_cl_{spectrum}"]
                )
            blocks.append(block)

        blocks.append(self.lensing["binmat_spt_lensing"] @ cl_kk[:3102])
        return np.concatenate(blocks)

    def create(
        self,
        dl_tt: np.ndarray,
        dl_te: np.ndarray,
        dl_ee: np.ndarray,
        cl_tt: np.ndarray,
        cl_te: np.ndarray,
        cl_ee: np.ndarray,
        cl_bb: np.ndarray,
        cl_pp: np.ndarray,
        calibration: Calibration = Calibration(),
    ) -> dict[str, np.ndarray]:
        """Return every smooth data vector in one dictionary."""
        result = self.primary_cmb(dl_tt, dl_te, dl_ee, calibration)
        result["lensing"] = self.lensing_cmb(cl_pp, cl_tt, cl_te, cl_ee, cl_bb)
        return result

    def coordinates(self) -> dict[str, np.ndarray]:
        """Return the effective multipoles associated with each data-vector block."""
        return {
            "camspec_ell_tt": self.primary["camspec_ell_tt"],
            "camspec_ell_te": self.primary["camspec_ell_te"],
            "camspec_ell_ee": self.primary["camspec_ell_ee"],
            "act_effective_ell_tt": self.primary["act_effective_ell_tt"],
            "act_effective_ell_te": self.primary["act_effective_ell_te"],
            "act_effective_ell_ee": self.primary["act_effective_ell_ee"],
            "spt_effective_ell": self.primary["spt_effective_ell"],
            "lensing_bcents_act": self.lensing["bcents_act_lensing"],
            "lensing_bcents_planck": self.lensing["bcents_planck_lensing"],
            "lensing_bcents_spt": self.lensing["bcents_spt_lensing"],
        }
