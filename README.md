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

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_state_policy.py \
  --seed 500 --max-steps 200 

python scripts/run_vision_policy.py \
  --seed 500 --max-steps 200 

```

Optional: train the vision encoder/regressor.

```bash
python scripts/train_vision_features.py \
  --dataset models/pusht_cchi_v7_replay.zarr.zip \
  --output-dataset models/pusht_cchi_v7_replay_imgs_feature_epoch_200.zarr \
  --checkpoint-path models/vision_state_predictor_epoch_200.ckpt
```

Scripts automatically download missing datasets/checkpoints and expose further
CLI knobs—run with `--help` for the full list.

## State policy key parameters

```bash
python scripts/run_state_policy.py \
  --seed 502 \
  --max-steps 200 \
  --k-neighbors 5 \
  --action-horizon 8 \
  --obs-noise-std 0.01 \
  --fixed-lambda1 1.0 \
  --fixed-lambda2 1.0 \
  --action-smoothing 0.2 \
  --no-video
```

- `--max-steps`: upper bound on rollout length.
- `--k-neighbors`: number of demonstrations blended per query.
- `--action-horizon`: planned steps drawn from the database.
- `--obs-noise-std`: initial latent noise injected for robustness.
- `--fixed-lambda1` / `--fixed-lambda2`: weights for progression vs. attraction flows.
- `--action-smoothing`: exponential filter applied to successive actions.

Omit any flag to fall back to the defaults in `run_state_policy.py`.

## Notes

- Downloads land in `models/`.
- Datasets should match the `.zarr` schema from diffusion-policy.
- Vision policy expects a ResNet18 checkpoint produced by the included trainer.
- `--no-video` skips mp4 generation if `skvideo.io` is unavailable.

## License
MIT License
The code is provided as-is for research reproduction.
