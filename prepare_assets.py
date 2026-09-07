#!/usr/bin/env python3
"""Build the compact binary assets used by the Asimov-data generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_fiducial_spectra(corrections_dir: Path) -> dict[str, np.ndarray]:
    lensed = np.loadtxt(corrections_dir / "cosmo2017_10K_acc3_lensedCls.dat")
    ell = lensed[:, 0].astype(int)
    factor = 2.0 * np.pi / (ell * (ell + 1.0))

    result = {}
    for name, column in (("tt", 1), ("ee", 2), ("bb", 3), ("te", 4)):
        values = np.zeros(3000)
        keep = ell < 3000
        values[ell[keep]] = lensed[keep, column] * factor[keep]
        result[f"fid_cl_{name}"] = values

    potential = np.loadtxt(corrections_dir / "cosmo2017_10K_acc3_lenspotentialCls.dat")
    lens_ell = potential[:, 0].astype(int)
    fid_cl_kk = np.zeros(3000)
    keep = lens_ell < 3000
    fid_cl_kk[lens_ell[keep]] = potential[keep, 5] * (2.0 * np.pi) / 4.0
    result["fid_cl_kk"] = fid_cl_kk
    return result


def _compact_lensing_kernels(
    corrections_dir: Path,
    binning_dir: Path,
) -> dict[str, np.ndarray]:
    fiducial = _load_fiducial_spectra(corrections_dir)
    mask_l2 = np.ones(3000)
    mask_l2[:2] = 0.0

    configurations = {
        "act": {
            "binmat": "binmat_act.npy",
            "norm": "n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax4000.txt",
            "norm_derivative": "norm_correction_matrix_Lmin0_Lmax4000.npy",
            "n1_prefix": "N1der_",
            "n1_suffix": "_lmin600_lmax3000_full.npy",
        },
        "planck": {
            "binmat": "binmat_planck.npy",
            "norm": "PLANCK_n0mv_fiducial_lmin600_lmax3000_Lmin0_Lmax3000.txt",
            "norm_derivative": "P18_norm_correction_matrix_Lmin0_Lmax3000.npy",
            "n1_prefix": "N1_planck_der_",
            "n1_suffix": "_lmin100_lmax2048.npy",
        },
    }

    output: dict[str, np.ndarray] = dict(fiducial)
    output["binmat_spt_lensing"] = np.load(binning_dir / "binmat_spt.npy")
    output["bcents_act_lensing"] = np.load(binning_dir / "bcents_act.npy")
    output["bcents_planck_lensing"] = np.load(binning_dir / "bcents_planck.npy")
    output["bcents_spt_lensing"] = np.load(binning_dir / "bcents_spt.npy")

    for experiment, config in configurations.items():
        binmat = np.load(binning_dir / config["binmat"])
        fid_norm = np.loadtxt(corrections_dir / config["norm"])[1, :3000]
        safe_norm = fid_norm.copy()
        safe_norm[safe_norm == 0.0] = 1.0
        norm_weight = -2.0 * mask_l2 * fiducial["fid_cl_kk"] / safe_norm
        weighted_binmat = binmat * norm_weight[None, :]

        output[f"binmat_{experiment}_lensing"] = binmat

        n1_kk = np.load(
            corrections_dir / f"{config['n1_prefix']}KK{config['n1_suffix']}",
            mmap_mode="r",
        )
        output[f"kernel_{experiment}_kk"] = binmat @ n1_kk[:3000, :3000]

        norm_derivatives = np.load(corrections_dir / config["norm_derivative"], mmap_mode="r")
        for index, spectrum in enumerate(("tt", "ee", "bb", "te")):
            n1 = np.load(
                corrections_dir / f"{config['n1_prefix']}{spectrum.upper()}{config['n1_suffix']}",
                mmap_mode="r",
            )
            kernel = binmat @ n1[:3000, :3000]
            kernel += weighted_binmat @ norm_derivatives[index, :3000, :3000]
            output[f"kernel_{experiment}_{spectrum}"] = kernel

    return output


def build_assets(cmblite_dir: Path, lensing_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    camspec = cmblite_dir / "camspec_npipe_lite"
    camspec_ell = np.load(camspec / "ell.npy").astype(int)
    camspec_spec = np.load(camspec / "spectrum_id.npy").astype(int)
    camspec_cuts = ((30, 1500), (30, 1000), (30, 600))
    primary: dict[str, np.ndarray] = {}
    for spec_id, (name, (ell_min, ell_max)) in enumerate(
        zip(("tt", "te", "ee"), camspec_cuts, strict=True)
    ):
        mask = (camspec_spec == spec_id) & (camspec_ell >= ell_min) & (camspec_ell <= ell_max)
        primary[f"camspec_ell_{name}"] = camspec_ell[mask]

    act = cmblite_dir / "act_dr6_cmbonly"
    act_ell = np.load(act / "ell.npy")
    act_spec = np.load(act / "spectrum_id.npy").astype(int)
    act_mins = (1500, 1000, 600)
    act_ranges = ((0, 45), (45, 90), (90, 135))
    for spec_id, (name, ell_min, (start, stop)) in enumerate(
        zip(("tt", "te", "ee"), act_mins, act_ranges, strict=True)
    ):
        mask = (
            (act_spec[start:stop] == spec_id)
            & (act_ell[start:stop] >= ell_min)
            & (act_ell[start:stop] <= 6500)
        )
        primary[f"act_effective_ell_{name}"] = act_ell[start:stop][mask]
        primary[f"act_window_{name}"] = np.load(act / f"window_{name.upper()}_{start}_{stop}.npy")[
            mask
        ]
        primary[f"act_window_ell_{name}"] = np.load(
            act / f"window_ell_{name.upper()}_{start}_{stop}.npy"
        ).astype(int)

    spt = cmblite_dir / "spt3g_d1_tne_lite"
    primary["spt_theory_ell"] = np.load(spt / "theory_ell.npy").astype(int)
    primary["spt_effective_ell"] = np.load(spt / "effective_ell.npy")
    for name in ("tt", "te", "ee"):
        primary[f"spt_window_{name}"] = np.load(spt / f"window_{name.upper()}_lxl.npy")

    np.savez_compressed(output_dir / "primary_cmb_assets.npz", **primary)

    binning_dir = lensing_dir / "act_planck_spt3g_lensing" / "actplanckspt3g_baseline"
    lensing = _compact_lensing_kernels(lensing_dir / "like_corrs", binning_dir)
    np.savez_compressed(output_dir / "lensing_assets.npz", **lensing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmblite-dir",
        type=Path,
        required=True,
        help="Directory containing the exported CamSpec, ACT, and SPT arrays",
    )
    parser.add_argument(
        "--lensing-dir",
        type=Path,
        required=True,
        help="Directory containing the v1.2 lensing products and like_corrs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "cmb_asimov" / "data"
    )
    args = parser.parse_args()
    build_assets(args.cmblite_dir, args.lensing_dir, args.output_dir)


if __name__ == "__main__":
    main()
