#!/usr/bin/env python3
"""Run CAMB v2 and generate smooth CMB Asimov data vectors."""

from __future__ import annotations

import json
from pathlib import Path

import camb
import numpy as np
from camb import model

from cmb_asimov import CMBAsimovGenerator, Calibration


# User settings
OUTPUT = Path("asimov_dataset.npz")
LMAX = 9500

H0 = 67.36
OMBH2 = 0.02237
OMCH2 = 0.1200
TAU = 0.0544
AS = 2.1e-9
NS = 0.9649
W0 = -1.0
WA = 0.0


def setup_camb() -> camb.CAMBparams:
    """Configure CAMB v2 for lensed primary and lensing-potential spectra."""
    major = int(camb.__version__.split(".", maxsplit=1)[0])
    if major != 2:
        raise RuntimeError(f"CAMB v2 is required; imported CAMB {camb.__version__}")

    pars = camb.CAMBparams()
    pars.set_dark_energy(w=W0, wa=WA, dark_energy_model="ppf")
    pars.set_cosmology(
        H0=H0,
        ombh2=OMBH2,
        omch2=OMCH2,
        omk=0.0,
        mnu=0.06,
        num_massive_neutrinos=1,
        nnu=3.044,
        tau=TAU,
    )
    pars.InitPower.set_params(As=AS, ns=NS)
    pars.set_for_lmax(
        LMAX,
        lens_potential_accuracy=8,
        lens_output_margin=2050,
    )
    pars.set_matter_power(redshifts=[0.0], kmax=10.0, k_per_logint=130)
    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version="mead2020")
    pars.Accuracy.AccuracyBoost = 1.0
    pars.Accuracy.lAccuracyBoost = 1.2
    pars.Accuracy.lSampleBoost = 1.0
    pars.DoLateRadTruncation = False
    pars.min_l_logl_sampling = 6000
    return pars


def run_camb(pars: camb.CAMBparams, lmax: int) -> dict[str, np.ndarray]:
    """Return zero-based lensed D_ell, raw C_ell, and raw C_ell^{phi phi}."""
    results = camb.get_results(pars)
    dl = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK", raw_cl=False)
    cl = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK", raw_cl=True)
    lens = results.get_lens_potential_cls(lmax=lmax, raw_cl=True)
    return {
        "dl_tt": dl[:, 0],
        "dl_ee": dl[:, 1],
        "dl_bb": dl[:, 2],
        "dl_te": dl[:, 3],
        "cl_tt": cl[:, 0],
        "cl_ee": cl[:, 1],
        "cl_bb": cl[:, 2],
        "cl_te": cl[:, 3],
        "cl_pp": lens[:, 0],
    }


def main() -> None:
    if LMAX < 8501:
        raise ValueError("LMAX must be at least 8501 for the ACT bandpower windows")

    pars = setup_camb()
    spectra = run_camb(pars, LMAX)
    generator = CMBAsimovGenerator()
    asimov = generator.create(
        spectra["dl_tt"],
        spectra["dl_te"],
        spectra["dl_ee"],
        spectra["cl_tt"],
        spectra["cl_te"],
        spectra["cl_ee"],
        spectra["cl_bb"],
        spectra["cl_pp"],
        Calibration(),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    coordinates = {f"coord_{name}": values for name, values in generator.coordinates().items()}
    np.savez_compressed(OUTPUT, **spectra, **asimov, **coordinates)
    metadata = {
        "camb_version": camb.__version__,
        "cosmology": {
            "H0": H0,
            "ombh2": OMBH2,
            "omch2": OMCH2,
            "tau": TAU,
            "As": AS,
            "ns": NS,
            "w0": W0,
            "wa": WA,
        },
        "lmax": LMAX,
        "primary_units": "D_ell in microkelvin^2",
        "lensing_primary_units": "raw lensed C_ell in microkelvin^2",
        "pp_units": "raw dimensionless C_ell^{phi phi}",
        "data_vector_order": {
            "camspec": "TT, TE, EE with NERSC cuts",
            "act": "TT, TE, EE with NERSC cuts",
            "spt": "TT(52), TE(72), EE(72), full lite likelihood",
            "lensing": "ACT(10), Planck(9), SPT(16)",
        },
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"CAMB {camb.__version__}")
    for name in ("camspec", "act", "spt", "lensing"):
        print(f"{name}: {asimov[name].shape}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
