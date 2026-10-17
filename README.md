# Unitree Go2 locomotion: wider velocity commands

NewUU robotics internship Senior Track submission. The experiment compares default flat-terrain velocity tracking with wider X/Y speed commands at two training budgets.

**Result:** the baseline learns accurate tracking; widening both linear ranges to ±2 m/s produces high tracking error. Increasing training from 300 to 900 iterations does not fix it in these runs.

- [Short report](report.md)
- [Original prediction](prediction.md)
- [Learning curves](figures/learning_curves.png)
- [First 300 iterations](figures/first_300_comparison.png)
- [Baseline recording](videos/baseline.mp4)
- [Wide-300 recording](videos/wide_300.mp4)
- [Wide-900 recording](videos/wide_900.mp4)

## Environment and evidence

Runs were performed in Ubuntu 24.04 under WSL on an NVIDIA RTX 4060 laptop GPU with 8 GB VRAM. Training used Newton MJWarp and RSL-RL PPO, 4,096 parallel environments, seed 42, and 24 rollout steps per iteration. All policies started fresh (`resume: false`).

Isaac Lab commit: `28a37cecdd433c22d9eabd6a5954add9f13a8951` (checkout originally named `v3.0-beta2`). The exact commit is the reproducibility anchor.

Observed setup: Python 3.12, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Newton 1.2.1, Warp 1.13.0, and RSL-RL 5.0.1. Saved `env.yaml` and `agent.yaml` files accompany each run. The exported evidence did not include a full dependency lock, so environment reconstruction is best effort; numerical determinism across GPUs is not claimed.

| Run directory | Budget | Final checkpoint |
|---|---:|---|
| `2026-10-13_09-28-07` | 300 | `model_299.pt` |
| `2026-10-13_10-21-07_wide_300` | 300 | `model_299.pt` |
| `2026-10-14_10-50-41_wide_900` | 900 | `model_899.pt` |

## Install the tested checkout

Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/) for OS and GPU prerequisites. The local project used the kitless Newton installation:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 28a37cecdd433c22d9eabd6a5954add9f13a8951

uv venv --python 3.12 env_isaaclab
source env_isaaclab/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
```

## Reproduce training

Run these commands from the Isaac Lab checkout with its environment activated. Keep the default environment count. New reproductions create new timestamped log directories.

Baseline, using the task's defaults:

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  physics=newton_mjwarp --visualizer none
```

Wider range, 300 iterations:

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --max_iterations 300 \
  physics=newton_mjwarp --visualizer none \
  'env.commands.base_velocity.ranges.lin_vel_x=[-2.0,2.0]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[-2.0,2.0]' \
  agent.run_name=wide_300
```

Wider range, 900 iterations, from scratch:

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --max_iterations 900 \
  physics=newton_mjwarp --visualizer none \
  'env.commands.base_velocity.ranges.lin_vel_x=[-2.0,2.0]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[-2.0,2.0]' \
  agent.resume=False \
  agent.run_name=wide_900
```

The wider-range runs change both linear command ranges; the longer run also changes the iteration budget. The source location and resolved configuration differences are documented in the report and `analysis/config_differences.json`.

## Play the submitted checkpoints

From the Isaac Lab checkout, replace `/absolute/path/to/submission` with the downloaded repository path:

```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --checkpoint /absolute/path/to/submission/checkpoints/baseline.pt \
  --num_envs 1 \
  physics=newton_mjwarp --visualizer newton
```

For either wider policy, use its checkpoint and preserve the wider playback commands:

```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --checkpoint /absolute/path/to/submission/checkpoints/wide_900.pt \
  --num_envs 1 \
  physics=newton_mjwarp --visualizer newton \
  'env.commands.base_velocity.ranges.lin_vel_x=[-2.0,2.0]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[-2.0,2.0]'
```

Replace `wide_900.pt` with `wide_300.pt` for the shorter wider run. Playback uses one environment; training used 4,096. These recordings demonstrate behavior and do not constitute held-out quantitative evaluation.

## Inspect and reproduce the comparison

TensorBoard can read the bundled original events:

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Open `http://localhost:6006`, select all three runs, and inspect `error_vel_xy`, `success_rate`, and `Episode_Termination/base_contact`. Smoothing zero shows the raw values.

To regenerate the CSV, JSON, configuration checks, and scientific plots, use a separate analysis environment from this repository root:

```bash
python3 -m venv .analysis_env
source .analysis_env/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/analyze.py
```

The script retains every scalar point, checks complete iteration coverage, rejects duplicates and nonfinite values, and validates the intended configuration changes. Late summaries average the final 50 log points. Raw values and trailing 20-point means are plotted without extending the shorter runs.

## Verify the files

The three final checkpoints are included in `checkpoints/`:

| File | Run | Final training iteration |
|---|---|---:|
| `baseline.pt` | Baseline, 300 iterations | 299 |
| `wide_300.pt` | Wider range, 300 iterations | 299 |
| `wide_900.pt` | Wider range, 900 iterations | 899 |

`submission_sha256.txt` contains checksums for the required submission files. From the repository root, verify their bytes with:

```bash
sha256sum -c submission_sha256.txt
```

The file-presence check can also be run with:

```bash
python scripts/check_submission.py
```

This script checks that required files are present and non-empty, then regenerates the checksum list. It does not evaluate the policies. [VALIDATION.md](VALIDATION.md) describes the data and file checks.
