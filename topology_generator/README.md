# EnvSched task-topology generator

Generate the `data/task_topologies.jsonl` used by the selector from existing EnvScaler scenarios and executable environment metadata. The extraction prompt (Appendix E of the paper) and topology validation follow the `task-topology-v2-agent-visibility` format used in the paper.

The inputs are scenario JSON (`task_id`, `env_id`, `env_class_name`, `task`, `init_config`, and `checklist_with_func`) and an environment metadata JSON object keyed by environment ID. Input metadata must be trusted: the extractor executes the provided verifier functions on a copy of each initial state. No rollout or reward history is used.

## Run

Run from the repository root with Python 3.10 or later:

```bash
python -m pip install -r topology_generator/requirements.txt
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='https://your-compatible-endpoint/v1'

python topology_generator/generate_topology.py \
  --scenarios /path/to/envscaler_rl_scenario_metadata.json \
  --env-metadata /path/to/environment_metadata.json \
  --model your-model-id \
  --output-dir outputs/topology \
  --dry-run
```

Remove `--dry-run` to call the model and generate labels. The endpoint must support streaming chat completions and JSON-object responses. Choose a model provided by your endpoint. `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1` if unset. You can pass `--api-key-file /path/to/key` instead of setting `OPENAI_API_KEY`.

`--workers 4` controls concurrency. `--task-id ID` (repeatable) and `--limit N` select a subset. `--stage tag` only fills the cache; `--stage export` exports validated cached labels without API calls. The default `--stage all` does both.

## Add new data

Append new scenarios with unique task IDs and provide metadata for their environments. Run the same command with the updated full scenario pool, the same model, and the same output/cache directory. The generator reuses matching cached labels and requests labels only for new or changed inputs. Successful tasks are cached even when another request fails, so rerunning resumes the remaining work.

The output `outputs/topology/task_topologies.jsonl` contains one record per selected task and can be used directly as the selector's `--topologies` input; copy it to `data/task_topologies.jsonl` for `train.py`. When training on an expanded pool, use the matching full scenario and environment files and regenerate topologies for that whole pool; the selector requires the task ID sets to match. `data/manifest.json` describes the paper's dataset and is not a checksum for regenerated or expanded data.

Other outputs are a CSV summary, extraction/quality audits, and a failure report. Model-generated labels may vary between runs. This tool annotates existing scenarios; creating new scenarios is handled by EnvScaler's `scen_generator`.

## Tests

The repository's `python -m pytest` runs these tests too. They use mocked model responses and do not call a paid API.
