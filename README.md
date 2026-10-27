# EnvSched

Code for **EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning**.

EnvSched decides which tasks to train on at each step of agentic RL. Instead of sampling tasks at random, it predicts which tasks will give informative (high-variance) rewards and spreads each batch across under-explored environments, using only the rollout history already collected. No extra pilot rollouts are needed.

## Contents

- The EnvSched scheduler (`selector/`)
- A GRPO training entry point for Qwen3-1.7B, 4B, and 8B thinking models on the EnvScaler pool (2,550 tasks in 51 environments), starting from the original models without SFT
- A modified, reduced copy of the [ROLL](https://github.com/alibaba/ROLL) training framework (`vendor/ROLL/`). An unmodified ROLL install will not work, because this copy adds curriculum updates, trajectory logging, and the EnvScaler environments.
- A tool that generates the task topologies the scheduler needs (`topology_generator/`)

## Quick start

### 1. Install

Training needs Linux x86_64, Python 3.10, NVIDIA GPUs, CUDA 12.4, cuDNN 9, and a C++/CUDA build toolchain. The pinned runtime is PyTorch 2.6.0, vLLM 0.8.4, and Megatron Core 0.16.1.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

Apex and Transformer Engine compile during installation. If needed, set `TORCH_CUDA_ARCH_LIST`, `NVTE_CUDA_ARCHS`, and `MAX_JOBS` for your machine.

To run only the scheduler, data preparation, and tests, a CPU environment is enough:

```bash
python -m pip install -r requirements-selector.txt 'pytest>=8,<10'
python -m pytest -q
```

### 2. Download the training data

```bash
python scripts/prepare_data.py
```

This downloads three files from a pinned EnvScaler commit, checks each against the SHA-256 in `data/manifest.json`, and places them where the training code expects them:

| File | Contents | Installed under |
| --- | --- | --- |
| `envscaler_rl_scenario_metadata.json` | 2,550 RL tasks | `vendor/ROLL/roll/pipeline/agentic/env/envscaler_env/data/` |
| `environment_metadata.json` | The 51 environments those tasks use, taken from EnvScaler's `191_env_metadata.json` | same as above |
| `data_multi_turn_base.json` | BFCL-v3 multi-turn base, used for validation during training | `vendor/ROLL/roll/pipeline/agentic/env/bfcl_env/data/` |

These are byte-for-byte the files used in the paper. Without network access, download them yourself and pass `--source-dir /path/to/EnvScaler` (a clone of the repository, or a folder holding the three files).

### 3. Generate the task topologies

The cross-environment edges need `data/task_topologies.jsonl`, one LLM-extracted topology per task. The prompt is in Appendix E of the paper. Any OpenAI-compatible chat endpoint works. Expect one request per task, 2,550 in total; finished tasks are cached, so an interrupted run resumes where it stopped.

```bash
python -m pip install -r topology_generator/requirements.txt
export OPENAI_API_KEY=...                         # your key
export OPENAI_BASE_URL=https://your-endpoint/v1   # only for non-OpenAI endpoints

DATA=vendor/ROLL/roll/pipeline/agentic/env/envscaler_env/data
python topology_generator/generate_topology.py \
  --scenarios $DATA/envscaler_rl_scenario_metadata.json \
  --env-metadata $DATA/environment_metadata.json \
  --model your-model-id \
  --output-dir outputs/topology
cp outputs/topology/task_topologies.jsonl data/
```

LLM output varies between models and runs, so your topologies will not match the paper's exactly. See `topology_generator/README.md` for more options.

### 4. Check everything before training

```bash
python scripts/prepare_data.py --check
python train.py --model-size 1.7b --run-dir runs/prepare --prepare-only
```

`--check` verifies the three data files and confirms the topologies cover exactly the 2,550 tasks. `--prepare-only` resolves the training config and picks the step-0 batch, without needing a GPU or model weights.

### 5. Train

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
python train.py --model-size 1.7b --model-path /path/to/Qwen3-1.7B --run-dir runs/qwen3-1.7b
```

| Option | Meaning |
| --- | --- |
| `--model-size` | `1.7b`, `4b`, or `8b`. Pipeline parallelism is 1 for 1.7B and 2 for 4B/8B, so 4B/8B need an even number of GPUs. |
| `--model-path` | Local directory of the original Qwen3 thinking model |
| `--run-dir` | Output directory. Use a new one for each run. |
| `--gpus` | GPU IDs to use; default `0,1,2,3,4,5,6,7` |
| `--max-steps` | Training iterations; default 201 (steps 0–200) |
| `--selector-python` | Python interpreter for the scheduler, if it lives in a separate CPU environment; default is the current one |

All three model sizes share one recipe: 64 tasks × 8 rollouts per step, synchronous rollout, learning rate `1e-6`, policy and advantage clip `0.2`, and no KL penalty. BFCL validation runs every step. Checkpoints are saved every 20 steps and at the last step.

## Where the outputs go

Inside `--run-dir`:

| Path | Contents |
| --- | --- |
| `selector/task_manifest.step_N.json` | The tasks picked for step N, with the reasons for each pick |
| `selector/input_audit.step_N.json` | What the scheduler read when picking step N |
| `output/cases/` | Rollout records, which the scheduler reads to pick the next step |
| `output/logs/` | Training logs |
| `checkpoint/` | Model checkpoints |
| `resolved_config.yaml` | The full training config actually used |

During training, ROLL runs `scripts/update_online_variance_curriculum.sh` after each step, and that script runs `selector/select_online_high_variance.py` to pick the next batch.

## Scheduler parameters

The defaults follow Appendix D of the paper. They are the same for every model size.

| Paper | Value | Where it is set |
| --- | --- | --- |
| Batch size B | 64 | `--batch-size` |
| Coverage share ρ | 0.5 (32 of 64 tasks) | `--least-observed-per-batch 32` |
| Exposure scale c₀ | 16 | `--exact-task-exposure-scale` |
| Similarity penalty λ_sim | 0.08 | `--diversity-penalty` |
| Per-step environment cap C₁ | 4 | `--max-per-environment` |
| Three-step environment cap C₃ | 11 | `--environment-window-steps 3 --max-per-environment-window 11` |
| Consecutive-selection cap C_task | 2 | `--max-consecutive-selections` |
| Recent-decline half-life h_short and weight η | 2, 0.5 | `--recency-guard-half-life`, `--recency-guard-strength` |
| Within- and cross-environment neighbors | 6, 12 | `--within-env-neighbors`, `--neighbors` |
| History half-life h_long | 8 | `--half-life-steps` |
| Graph and own-history shrinkage λ_g, λ_s | 2, 2 | `--graph-shrinkage`, `--own-history-shrinkage` |

`scripts/update_online_variance_curriculum.sh` passes the coverage, exposure, quota, penalty, and recent-decline values; the rest are defaults in `selector/select_online_high_variance.py`.

### Faster graph backend (optional)

By default the scheduler stores full N×N similarity matrices. For large task pools, add `--graph-execution-mode incremental_exact` to `curriculum_update_command` in `configs/selector.yaml`. The scheduler then keeps sparse features and per-task neighbor lists, and scores only newly observed tasks each step (Section 3.3.2 of the paper). It picks exactly the same tasks as the default; `tests/test_incremental_similarity_graph.py` checks this.

## Repository layout

| Path | Purpose |
| --- | --- |
| `selector/select_static_cold_start.py` | Picks the step-0 batch |
| `selector/select_online_high_variance.py` | Builds the graph, predicts variance, and picks each later batch |
| `selector/task_progress_retirement.py` | Progress filter |
| `selector/online_rewarded_id_hack_guard.py`, `selector/id_hack_replay.py` | Identifier filter, with replay of saved actions |
| `selector/reward_statistics.py`, `selector/task_features.py` | Reads rollout rewards; extracts static task and verifier features |
| `selector/incremental_similarity_graph.py` | Optional faster graph backend |
| `topology_generator/` | Generates the task topologies with an LLM |
| `train.py`, `configs/` | Training entry point and the three model profiles |
| `scripts/` | Installation, data download, and the per-step scheduler call |
| `vendor/ROLL/` | Modified ROLL runtime with the EnvScaler and BFCL environments |
| `python_patches/` | Compatibility patches for the training stack |
| `data/` | Data manifest and fixed sampling seeds; downloaded and generated data also go here |
| `tests/` | CPU tests |

## Limitations

- The bundled BFCL environment is for validation during training. It is not an official BFCL scoring suite.
- This release was tested with the CPU test suite, data download from EnvScaler, and `train.py --prepare-only`. A fresh CUDA installation and full GPU training were not re-run as part of packaging.

## License

MIT; see `LICENSE`. `vendor/ROLL` keeps its Apache-2.0 license in `vendor/ROLL/LICENSE`. See `THIRD_PARTY_NOTICES.md` for the data and other third-party components. Model weights and dependencies keep their own licenses.
