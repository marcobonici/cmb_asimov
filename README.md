# cmb_asimov

`cmb_asimov` generates deterministic, noise-free CMB data vectors for internal
tests and forecast pipelines. It runs CAMB v2, obtains smooth lensed primary-CMB
and lensing-potential spectra, and passes them through the same selections,
bandpower windows, calibrations, aberration correction, and lensing response
corrections used by the corresponding likelihoods.

The generated data are Asimov means,

$$
d_{\mathrm A}=m(\theta_{\mathrm fid}),
$$

with no random draw from any experimental covariance.

## Included likelihood mappings

- Planck PR4/NPIPE CamSpec lite:
  - TT: $30\leq\ell\leq1500$;
  - TE: $30\leq\ell\leq1000$;
  - EE: $30\leq\ell\leq600$.
- ACT DR6 CMB-only lite:
  - TT above the inclusive effective-multipole cut $\ell=1500$;
  - TE above $\ell=1000$;
  - EE above $\ell=600$.
- The complete SPT-3G D1 lite TT/TE/EE data vector.
- Joint ACT DR6 + Planck NPIPE + SPT-3G lensing, using the
  `actplanckspt3g_baseline` configuration with active normalization and $N_1$
  corrections.

Planck low-TT and SROLL2 are deliberately excluded. Their released products
encode non-Gaussian observed likelihoods rather than reusable linear
spectrum-to-data mappings, so pure input spectra do not uniquely define
replacement likelihood data.

## Installation

CAMB v2 and NumPy are the only runtime dependencies:

```bash
pip install .
```

The package checks at runtime that the imported CAMB major version is 2.

## Generate a dataset

Edit the `User settings` block at the top of `generate_asimov.py`:

```python
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
```

Then run:

```bash
python generate_asimov.py
```

The output `.npz` contains:

- lensed `dl_tt`, `dl_te`, `dl_ee`, and `dl_bb`;
- raw lensed `cl_tt`, `cl_te`, `cl_ee`, and `cl_bb`;
- raw lensing-potential `cl_pp`;
- smooth `camspec`, `act`, `spt`, and `lensing` data vectors;
- effective multipoles and bin centers under `coord_*` keys.

A JSON sidecar records the CAMB version, cosmological parameters, units, and
data-vector ordering. `example_asimov_dataset.npz` and its JSON sidecar provide
a complete output generated with CAMB 2.0.0 and the settings above.

## Spectrum conventions

Every array starts at multipole zero, including explicit entries for
$\ell=0,1$.

The primary-CMB likelihoods consume lensed

$$
D_\ell^{XY}=\frac{\ell(\ell+1)}{2\pi}C_\ell^{XY}
$$

in $\mu\mathrm{K}^2$. CAMB supplies these through
`get_lensed_scalar_cls(CMB_unit="muK", raw_cl=False)`.

The lensing response corrections instead consume raw lensed TT, TE, EE, and BB
$C_\ell$ in $\mu\mathrm{K}^2$, together with raw dimensionless
$C_L^{\phi\phi}$. The latter is converted to convergence through

$$
C_L^{\kappa\kappa}
=\frac{[L(L+1)]^2}{4}C_L^{\phi\phi}.
$$

The generating calibration parameters default to unity. Non-unit values can be
supplied through the `Calibration` dataclass.

## Binary assets

The required numerical assets are committed directly in the repository:

| File | Contents | Size |
|---|---|---:|
| `cmb_asimov/data/primary_cmb_assets.npz` | CamSpec selections and ACT/SPT windows | 5.31 MiB |
| `cmb_asimov/data/lensing_assets.npz` | Lensing binning matrices and compact response kernels | 1.25 MiB |

The compact lensing kernels are the original normalization and $N_1$ response
matrices contracted with the final ACT and Planck binning matrices. They are
algebraically equivalent to the multi-gigabyte unpacked response matrices. A
direct comparison with the full calculation gave a maximum relative difference
of $2.99\times10^{-15}$.

Checksums for the packaged assets and example output are recorded in
`SHA256SUMS`.

## Original public sources

The compact files were prepared from the following released likelihood
products.

### CamSpec NPIPE lite

- Code: https://github.com/HTJense/camspec_npipe-lite
- Data:
  https://github.com/HTJense/camspec_npipe-lite/releases/download/v1.0/CamSpec_NPIPE_cmb_sacc.tar.gz
- Archive SHA-256:
  `657814ff6d14642737dbdfb5e7ff3e1c570ce9b237b95c5695bb19a38a4dfec7`

### ACT DR6 CMB-only

- Code: https://github.com/ACTCollaboration/DR6-ACT-lite
- Data:
  https://lambda.gsfc.nasa.gov/data/act/pspipe/sacc_files/dr6_data_cmbonly.tar.gz
- Archive SHA-256:
  `3f057c2569211ada03759530b74848b322edc3d68d66b6b8c2db0679547dbbd8`

The LAMBDA archive contains the real foreground-marginalized
`v1.0/dr6_data_cmbonly.fits`; the smaller SACC file bundled in the code
repository is a simulated test product.

### SPT-3G D1 T&E lite

- Official data repository:
  https://github.com/SouthPoleTelescope/spt_candl_data
- Source directory:
  `spt_candl_data/SPT3G_D1_TnE_v0/lite/`

This directory contains the released bandpowers, covariance, likelihood YAML,
and TT/TE/EE window functions.

### Joint ACT + Planck + SPT lensing

- Combined likelihood code and base products:
  https://github.com/qujia7/spt_act_likelihood
- Official LAMBDA code archive:
  https://lambda.gsfc.nasa.gov/data/suborbital/act_spt_joint/spt_act_likelihood-1.0.tar.gz
- ACT DR6 v1.2 fiducial spectra and correction matrices:
  https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/ACT_dr6_likelihood_v1.2.tgz

The committed binary files are sufficient for normal use. `prepare_assets.py`
is a maintainer utility for rebuilding them from prepared exports of these
source products; users do not need to run it.

## Validation

The implementation has been checked against the released products:

- ACT bandpowers agree exactly with direct SACC window application;
- all packaged SPT windows agree element-by-element with the official files;
- CAMB's $D_\ell/C_\ell$ and $\phi\phi/\kappa\kappa$ conversions agree to
  floating-point precision;
- the compact lensing response agrees with the original full response matrices
  to machine precision.

Run the test suite with:

```bash
python -m unittest discover -s tests
```
