# GeoCondSR

Code for the paper **"Three-dimensional super-resolution of sandstone micro-CT images using a geologically conditioned generative network"** (manuscript under review).

GeoCondSR reconstructs 2-μm pore structure from 14-μm micro-CT scans of sandstone. The network has three parts:

- a deterministic **mean path** that places pores and grains from the coarse scan;
- a **residual texture generator** that adds the detail the coarse scan cannot determine, modulated by sample-level geological state parameters computed from fine-scan gray-level statistics;
- a **hard projection** through the scanner degradation operator measured from real image pairs, which keeps the output consistent with the coarse scan.

The models were trained and evaluated on real paired 14-μm and 2-μm scans of 14 sandstones from seven regions, with leave-one-region-out cross-validation, and applied to whole-plug scans of 34 core plugs.

## Repository layout

```
src/            network definitions, data loading and training scripts
  models.py, models54.py     generator, discriminator, mean path, geological-state predictor
  models59.py                SwinIR-3D and 3D diffusion baselines
  dataset.py                 paired-block dataset and fold definitions
  train.py                   generator pre-training and loss functions
  train54.py                 mean path and final generator training
  train59n.py                degradation-aware fine-tuning for whole-plug scans
  train_edsr.py, train_srgan.py, train59.py, train_cpred.py   baselines and geological-state predictor
  lbm59.py, lbm59s.py        D3Q19 two-relaxation-time lattice Boltzmann solver
  paths.py                   data, checkpoint and output locations
preprocessing/  registration of the dual-resolution scans, gray-level normalization,
                degradation kernel, geological state parameters, export of paired blocks
experiments/    evaluation scripts, one folder per part of the paper (each has run.sh)
figures/        scripts that draw the figures and tables from the evaluation outputs
configs/        fold definitions, sample properties and the configuration of every training run
scripts/        train_from_config.py and train_all.sh
```

## Installation

Tested with Python 3.12, PyTorch 2.10 (CUDA 12.8), NumPy 1.26, SciPy 1.13, scikit-image 0.25 and Matplotlib 3.10 on Linux with NVIDIA A100 (80 GB) GPUs.

```bash
conda create -n geocondsr python=3.12
conda activate geocondsr
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data and pretrained models

The paired dataset, the whole-plug context windows and the trained models are archived on Zenodo (DOI to be added). Download them and either place them in `data/` and `runs/` inside the repository or point the environment variables to their location:

```bash
export GEOCOND_DATA=/path/to/data      # paired blocks and derived inputs
export GEOCOND_RUNS=/path/to/runs      # model checkpoints
export GEOCOND_OUT=/path/to/outputs    # evaluation outputs (created automatically)
```

Expected layout of `GEOCOND_DATA`:

```
pairs/G??/*.npz          paired blocks: 36^3 coarse context (14 um) and 112^3 fine target (2 um)
meta/                    fold splits and registration metadata
cfield/, cfield2/        geological state parameters at every grid site
arr/, arr2/, cfield_arr/ auxiliary per-site arrays
degradation/H_kernel.npz measured degradation kernel (9^3)
cache/                   registered coarse-scan volumes (coarse-scan statistics)
plugs/                   whole-plug context windows, plug table, calibration parameters, noise spectra
```

Each run folder in `GEOCOND_RUNS` (for example `CQ_g54b/`) holds the checkpoint used in the paper. Run names combine the test region of the fold (CQ, SC, YN, GZ, SD, SHX) and the model:

| Suffix | Model |
|---|---|
| `m54b` | mean path |
| `n54b` | residual texture generator, geology-free (training-set mean state) |
| `g54b` | residual texture generator, geology-conditioned |
| `g54c`, `g54n`, `g54a` | generators trained with the coarse-scan state, the regressed neighbouring state and the state around the target block |
| `cpred` | geological-state predictor from the coarse scan |
| `edsr`, `srgan`, `swin`, `diff` | EDSR-3D, SRGAN-3D, SwinIR-3D and 3D diffusion baselines |
| `m59n`, `n59n`, `g59n` | degradation-aware fine-tuned models for whole-plug scans |
| `proj`, `rot2` | generator pre-training stages (initialization only) |

## Training

Every training run of the paper is recorded in `configs/runs/`. To repeat one run:

```bash
python scripts/train_from_config.py CQ_g54b --gpu 0
```

`bash scripts/train_all.sh 0` trains all models of all folds in dependency order.

## Reproducing the results

Run the experiment folders in this order; each `run.sh` loops over the six folds (set `GPU=` to choose the device).

| Folder | Content of the paper |
|---|---|
| `experiments/03_baselines` | comparison with trilinear interpolation, EDSR-3D, SRGAN-3D, SwinIR-3D and the diffusion model |
| `experiments/05_texture_and_microstructure` | detail texture, 3D pore structure, microstructure statistics, control of the output by the geological state |
| `experiments/01_main_evaluation` | sources of the geological state |
| `experiments/02_geological_state` | spatial structure of the state, sparse fine-scan calibration, prediction of the sample-level state |
| `experiments/04_flow_simulation` | lattice Boltzmann permeability and percolation |
| `experiments/06_whole_plug` | imaging-condition calibration and whole-plug application |
| `figures/` | figures and tables drawn from the aggregated outputs |

The scripts in `preprocessing/` start from the raw TIFF slices and rebuild the released dataset: registration in `preprocessing/registration/`, then `hkernel.py`, `cfield2.py`, `export_pairs.py` and `pack_for_gpu.py`.

## License

The code is released under the MIT License (see `LICENSE`). The dataset is released separately under CC BY 4.0.

## Citation

If you use this code or the dataset, please cite the paper above. Full bibliographic details will be added after publication.
