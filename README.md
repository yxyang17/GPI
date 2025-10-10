# PushT GPI Policies

Minimal implementation of Geometry-Aware Policy Imitation (GPI) for the PushT
environment. Core GPI components live under `gpi/`, while PushT datasets,
environments, and evaluation helpers sit in `pusht/`.

## Project layout

```
gpi/          shared database, planner, and policy logic
pusht/        PushT datasets, environments, downloads, evaluation
scripts/      entry points for running and training policies
requirements.txt
```


### Installation

```bash
conda env create -f environment.yml
conda activate gpi
```
Troubleshoots
There might be a version mismatch between pygame pymunk zarr and gym. Make sure install the correct version. 


## Quick start
```bash
python scripts/run_state_policy.py \
  --seed 500 --max-steps 200

python scripts/run_vision_policy.py \
  --seed 500 --max-steps 200
```

Optional: train the vision encoder/regressor.

```bash
conda activate gpi
python scripts/train_vision_features.py \
  --dataset models/pusht_cchi_v7_replay.zarr.zip \
  --output-dataset models/pusht_cchi_v7_replay_imgs_feature_epoch_200.zarr \
  --checkpoint-path models/vision_state_predictor_epoch_200.ckpt
```

Scripts automatically download missing datasets/checkpoints and expose further
CLI knobs—run with `--help` for the full list.

## Policy CLI parameters

`scripts/run_state_policy.py` accepts a rich set of knobs so you can reproduce our experiments or explore new settings:

- `--dataset`: input PushT replay archive (auto-downloaded into `models/`).
- `--seed`: environment RNG seed.
- `--max-steps`: hard cap on rollout length.
- `--k-neighbors`: number of demonstrations blended per query state.
- `--action-horizon`: number of steps fetched from the database per solve.
- `--subset-size`: use a random subset of the database for faster lookups.
- `--batch-size`: PyTorch batch size used during nearest-neighbour scoring.
- `--device`: override the default `cuda`/`cpu` selection.
- `--obs-noise-std`, `--noise-decay`, `--min-noise-std`, `--disable-noise`: control the stochastic exploration injected into latent states.
- `--random-seed`: RNG seed for subset sampling and policy noise.
- `--memory-length`: truncate the visited-state memory used to prevent loops.
- `--fixed-lambda1`, `--fixed-lambda2`: mix the progression and attraction flows with custom weights.
- `--action-smoothing`: exponential smoothing over the executed actions.
- `--no-relative-action`: operate directly in absolute action space.
- `--video-path`, `--no-save-video`: store or disable the mp4 rollout video (defaults to saving under `results/`).
- `--no-live-render`: skip the interactive window if you are running headless.
- `--quiet`: silence the tqdm progress indicator.

`scripts/run_vision_policy.py` mirrors the same interface and adds vision-specific flags:

- `--vision-checkpoint`: path to the ResNet18 encoder/regressor checkpoint.
- `--dynamic-subset` / `--subset-change-interval`: periodically refresh the KNN subset.
- `--memory-length`: positive integer to limit the loop-avoidance buffer (defaults to 500 for vision).

All parameters have sensible defaults, so you can omit any flag you do not need.

## Batch evaluation scripts

Use `scripts/run_state_evaluation.py` and `scripts/run_vision_evaluation.py` to sweep many random configurations:

- `--dataset` (both) and `--checkpoint` (vision): source assets for evaluation.
- `--count`: number of random runs to generate.
- `--max-steps`: rollout horizon for each run.
- `--random-seed`: seed the configuration generator for repeatability.
- `--no-save-video`: turn off mp4 export (enabled by default); pair with `--video-dir` to control the output folder.

Each run prints `Reward`, `Inference Time`, and `Memory` on a single tab-aligned line and logs the same data (plus any video path) under `results/logs/`.

## Notes

- Downloads land in `models/`.
- Datasets should match the `.zarr` schema from diffusion-policy.
- Vision policy expects a ResNet18 checkpoint produced by the included trainer.
- `--no-save-video` skips mp4 generation if `skvideo.io` is unavailable.

## License
MIT License
The code is provided as-is for research reproduction.
