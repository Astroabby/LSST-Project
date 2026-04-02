This work was conducted under the supervision of Agnès Ferté and Andrés A. Plazas Malagón.

# LSST Y1 Numerical Resolution Study (CosmoSIS)

This repository contains the CosmoSIS configuration files and pipeline setup used in a numerical resolution study for LSST Year 1 3x2pt analyses.

The purpose of this work is to quantify how numerical resolution settings affect the calculation of theoretical angular power spectra, the resulting cosmological parameter inference, and the computational cost of the analysis. The final goal is to identify numerical configurations that provide a good balance between accuracy and efficiency.


## Repository structure

The repository is organized into two main cases:

### Case_1
- Intrinsic Alignments: NLA model
- Galaxy Bias: Linear bias

### Case_2
- Intrinsic Alignments: TATT model
- Galaxy Bias: Non-linear bias (FAST-PT)

Each case contains the configuration files used to generate the simulated LSST Y1 data vector and perform the analysis for both modeling scenarios:

- `lsst_simulate.ini`: generates the data vector
- `lsst_analyze.ini`: computes theoretical predictions and likelihood
- `lssty10_32pt_simulate_values.ini`: cosmological and nuisance parameters


## Numerical parameters studied

The following resolution parameters were varied in this work:

### Case 1 (CAMB + projection)
- `n_k` (wavenumber sampling)
- `n_z` (redshift sampling)
- `n_ell` (multipole sampling)

### Case 2 (FAST-PT)
- `k_res_fac` (internal k-grid resolution)

The fiducial values included in the `.ini` files correspond to the highest resolution configuration used to generate the simulated data vector.


## Other configurations (optimized and low-resolution)

The optimized and low-resolution ("bad") configurations discussed in the technical note are obtained by modifying only the numerical resolution parameters:

### Case 1
- Optimized: `n_k = 100`, `n_z = 15`, `n_ell = 70`
- Low-resolution: `n_k = 20`, `n_z = 5`, `n_ell = 20`

### Case 2
- Optimized: `k_res_fac = 0.7`
- Low-resolution: `k_res_fac = 0.01`

All other parameters remain unchanged.

## How to run

### 1. Generate simulated data vector

`cosmosis examples/Case_1/lsst_simulate.ini`

### 2. Obtain theoretical prediction

`cosmosis examples/Case_1/lsst_analyze.ini`

## Parameter inference
The analysis was performed using:

- A test sampler (for likelihood evaluation)
- A grid sampler (for parameter estimation)

To switch between samplers, modify `sampler = test` to `sampler = grid` and specify the output settings:
[output]
save_dir = output/Case
filename = output/Case/grid_.txt
format = text
lock = F

## Notes

- The covariance matrix is fixed across all runs.
- Photometric redshift bias and shear calibration parameters are set to zero in this baseline analysis.
- Only $\Omega_m$ and $\sigma_8$ are varied in the parameter inference.
