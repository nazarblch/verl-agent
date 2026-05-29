# ARC-AGI-3 Environment for verl-agent

This environment integrates **ARC-AGI-3** into `verl-agent` as an interactive program-execution environment.

Unlike classic ARC grid tasks, ARC-AGI-3 is treated as a **turn-based game API**. The model writes a Python policy, the environment executes actions in the ARC runtime, monitors expected vs actual outcomes, and feeds compact monitor reports back into the next prompt for reflection.

## Overview

The solving loop is:

```text
[1. INFERENCE / LLM]
    Generate a hypothesis, plan, action or action_sequence, and expected outcome(s).
        ↓
[2. ACTION / Environment]
    Execute one ARC-AGI-3 action, or a controlled short action_sequence.
        ↓
[3. COMPARISON / Monitor]
    Compare expected state/visual predicates with the actual next Frame.
        ↓
[4. REFLECTION / LLM]
    Next prompt receives compact monitor summaries so the model can update its working memory.
```

Main features:

- Official ARC-AGI-3 interactive runtime via `arc_agi.Arcade().make(game_id)`.
- Python policy execution in an isolated subprocess.
- Single-action and controlled multi-action modes.
- Per-step monitor reports.
- Frame-diff checker and visual predicate DSL.
- Object, color, cell, and coordinate checks.
- Compact context/history compression for long episodes.
- Legacy/static ARC grid fallback mode.

---

## Files

| File | Purpose |
|---|---|
| `envs.py` | Main batched ARC-AGI-3 environment, official runtime wrapper, grid fallback, monitor, visual DSL, context compression |
| `projection.py` | Extracts Python code from model output |
| `__init__.py` | Exports environment builder and projection function |
| `../../prompts/arc_agi_3.py` | Prompt templates for official and grid modes |
| `../../env_manager.py` | `ArcAGI3EnvironmentManager` integration and memory handling |
| `../../../../verl/trainer/config/ppo_trainer.yaml` | `env.arc_agi_3` config defaults |
| `../../../../tests/environments/test_arc_agi_3_env.py` | Environment, monitor, visual DSL, and sequence tests |

---

## Installation

From the repository root:

```bash
cd /a0/usr/workdir/verl-agent
pip install -e .
pip install -e .[arc_agi_3]
```

If the extra does not install the ARC package, install it explicitly:

```bash
pip install 'arc-agi>=0.9.1'
```

For full training, the normal `verl` training stack is also required: `torch`, `ray`, CUDA, and a rollout backend such as `vllm` or `sglang`.

---

## ARC-AGI-3 Runtime Configuration

For official online ARC-AGI-3 runtime, configure environment variables according to the official ARC-AGI-3 agents repository. A typical setup is:

```bash
export ARC_API_KEY='...'
export ARC_BASE_URL='https://three.arcprize.org/'
export OPERATION_MODE='online'
```

For local/offline ARC runtime modes, use the variables required by the official ARC-AGI-3 runtime.

---

## Dataset Format

`verl-agent` expects parquet files. Each row should contain `env_kwargs` describing the game.

Minimal official-mode row:

```json
{
  "data_source": "arc_agi_3",
  "prompt": [],
  "ability": "arc_agi_3",
  "reward_model": {"style": "rule", "ground_truth": null},
  "extra_info": {"index": 0, "game_id": "ls20"},
  "env_kwargs": {
    "mode": "official",
    "game_id": "ls20",
    "data_source": "arc_agi_3"
  }
}
```

Create small train/val parquet files:

```bash
cd /a0/usr/workdir/verl-agent
mkdir -p /a0/usr/workdir/arc_agi_3_data

python - <<'PY'
import pandas as pd
from pathlib import Path

games_train = ["ls20"]
games_val = ["ls20"]

out_dir = Path('/a0/usr/workdir/arc_agi_3_data')
out_dir.mkdir(parents=True, exist_ok=True)

def make_rows(games):
    rows = []
    for i, game_id in enumerate(games):
        rows.append({
            'data_source': 'arc_agi_3',
            'prompt': [],
            'ability': 'arc_agi_3',
            'reward_model': {'style': 'rule', 'ground_truth': None},
            'extra_info': {'index': i, 'game_id': game_id},
            'env_kwargs': {
                'mode': 'official',
                'game_id': game_id,
                'data_source': 'arc_agi_3',
            },
        })
    return rows

pd.DataFrame(make_rows(games_train)).to_parquet(out_dir / 'train.parquet')
pd.DataFrame(make_rows(games_val)).to_parquet(out_dir / 'val.parquet')

print(out_dir / 'train.parquet')
print(out_dir / 'val.parquet')
PY
```

For multiple games:

```python
games_train = ["ls20", "ls21", "ls22"]
games_val = ["ls20"]
```

---

## Basic Training Command

A minimal ARC-AGI-3 training/evaluation command starts with:

```bash
cd /a0/usr/workdir/verl-agent

python -m verl.trainer.main_ppo \
  data.train_files=/a0/usr/workdir/arc_agi_3_data/train.parquet \
  data.val_files=/a0/usr/workdir/arc_agi_3_data/val.parquet \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  env.env_name=arc_agi_3 \
  env.arc_agi_3.mode=official \
  env.arc_agi_3.game_id=ls20 \
  env.max_steps=80 \
  env.history_length=2 \
  env.rollout.n=1 \
  env.arc_agi_3.require_program=true \
  env.arc_agi_3.program_timeout=5.0 \
  env.arc_agi_3.program_memory_mb=512 \
  env.arc_agi_3.max_action_sequence_len=1 \
  env.arc_agi_3.context_compression=true \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1
```

This usually still needs model and rollout backend overrides.

Example with a small Qwen model and vLLM:

```bash
cd /a0/usr/workdir/verl-agent

python -m verl.trainer.main_ppo \
  data.train_files=/a0/usr/workdir/arc_agi_3_data/train.parquet \
  data.val_files=/a0/usr/workdir/arc_agi_3_data/val.parquet \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  env.env_name=arc_agi_3 \
  env.arc_agi_3.mode=official \
  env.arc_agi_3.game_id=ls20 \
  env.max_steps=80 \
  env.history_length=2 \
  env.rollout.n=1 \
  env.arc_agi_3.require_program=true \
  env.arc_agi_3.program_timeout=5.0 \
  env.arc_agi_3.program_memory_mb=512 \
  env.arc_agi_3.max_action_sequence_len=1 \
  env.arc_agi_3.context_compression=true \
  env.arc_agi_3.max_history_monitor_mismatches=4 \
  env.arc_agi_3.max_frame_changed_cells=64 \
  env.arc_agi_3.max_prompt_frame_cells=256 \
  trainer.project_name=arc_agi_3 \
  trainer.experiment_name=arc_agi_3_single_action_smoke \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1
```

Adjust backend-specific arguments for `sglang`, FSDP, Megatron, etc. according to existing `verl` examples.

---

## Recommended Curriculum

Start conservatively, then enable more planning.

### Stage 1: Single-action exploration

```bash
env.arc_agi_3.max_action_sequence_len=1 \
env.max_steps=80 \
env.history_length=2
```

Use this for early RL. It gives cleaner credit assignment because each LLM turn produces one action.

### Stage 2: Short controlled sequences

```bash
env.arc_agi_3.max_action_sequence_len=2 \
env.arc_agi_3.stop_sequence_on_mismatch=true
```

Use once the model reliably emits valid Python policies.

### Stage 3: Controlled planning

```bash
env.arc_agi_3.max_action_sequence_len=3
```

Use when `expectation_met` is frequently true and visual predictions are meaningful.

### Stage 4: Evaluation/exploitation

```bash
env.arc_agi_3.max_action_sequence_len=3
# or cautiously 5
```

Avoid large sequence lengths until the model reliably predicts dynamics.

---

## Python Policy Contract

The model must answer with executable Python inside tags:

```text
<think>
Reason about the frame, hypothesis, and expected effect.
</think>
<python>
def choose_action(context):
    ...
</python>
```

Supported contracts:

```python
def choose_action(frames, latest_frame): ...
def choose_action(context): ...
def solve(context): ...
action = {...}
answer = {...}
```

The environment executes the Python policy in an isolated subprocess and expects one action or an `action_sequence`.

### Single action example

```python
def choose_action(context):
    latest = context["latest_frame"]

    if latest.get("state") in ("NOT_PLAYED", "GAME_OVER"):
        return {
            "action": "RESET",
            "hypothesis": "The game is not started.",
            "plan": "Reset to obtain a playable frame.",
            "expected_state": "NOT_FINISHED",
            "expected_level_delta": 0,
            "expected_win": False,
        }

    return {
        "action": "ACTION1",
        "hypothesis": "Test whether ACTION1 moves or transforms the main object.",
        "plan": "Execute ACTION1 and verify frame change.",
        "expected_state": "NOT_FINISHED",
        "expected_frame_changed": True,
        "expected_changed_cell_count": {"gte": 1},
    }
```

### Controlled multi-action example

```python
def choose_action(context):
    return {
        "hypothesis": "The avatar must move right and activate the target.",
        "plan": "Move right twice, then interact if each movement is confirmed.",
        "action_sequence": [
            {"action": "ACTION4", "expected_frame_changed": True},
            {"action": "ACTION4", "expected_frame_changed": True},
            {"action": "ACTION5", "expected_win": True},
        ],
        "stop_on_mismatch": True,
    }
```

The environment executes the sequence step by step and stops early on mismatch, invalid action, terminal state, max steps, or max sequence length.

---

## Actions

| Action | Meaning |
|---|---|
| `RESET` | Start or restart the game |
| `ACTION1` | Simple action 1, commonly Up / W |
| `ACTION2` | Simple action 2, commonly Down / S |
| `ACTION3` | Simple action 3, commonly Left / A |
| `ACTION4` | Simple action 4, commonly Right / D |
| `ACTION5` | Main interaction, commonly Enter / Space / Delete |
| `ACTION6` | Click/point action with `x`, `y` in `0..63` |
| `ACTION7` | Undo, if available |

`ACTION6` example:

```python
{
    "action": "ACTION6",
    "x": 32,
    "y": 45,
    "hypothesis": "The target object is clickable at this coordinate.",
    "plan": "Click the likely target and verify level completion.",
    "expected_level_delta": 1,
    "expected_win": True,
}
```

---

## Visual DSL and Frame-Diff Monitoring

The monitor checks scalar and visual expectations.

### Scalar expectations

| Field | Compared with |
|---|---|
| `expected_state` | Actual `state` |
| `expected_level_delta` | Delta of `levels_completed` |
| `expected_levels_completed` | Actual `levels_completed` |
| `expected_win` | Actual `won` |

### Visual expectations

| Field | Meaning |
|---|---|
| `expected_frame_changed` | Whether the frame changed |
| `expected_changed_cell_count` | Number of changed cells, int or `{ "eq"/"gte"/"lte": ... }` |
| `expected_changed_cells` | Specific cells expected to change from value to value |
| `expected_cell` | Expected color/value at a coordinate |
| `expected_cells` | Multiple cell checks |
| `expected_coordinate` | Alias for `expected_cell` |
| `expected_color_count` | Count of one color before/after |
| `expected_color_counts` | Multiple color-count checks |
| `expected_color_delta` | Change in count for one color |
| `expected_color_deltas` | Multiple color-delta checks |
| `expected_colors_after` | Expected color histogram after action |
| `expected_object_moved` | Connected component of a color moved by `dx`, `dy` |
| `expected_object_at` | Object exists at a bbox |
| `expected_objects` | Multiple object checks |
| `expected_visual` | One predicate or predicate list |
| `expected_visual_predicates` | List of safe visual predicates |

### Predicate operations

Supported safe predicate ops:

- `changed`
- `unchanged`
- `cell_equals`
- `cell_changed`
- `color_count`
- `color_count_delta`
- `changed_cell_count`
- `object_moved`
- `object_exists`
- `all`
- `any`
- `not`

No user predicate is executed as Python code; invalid predicates become monitor mismatches.

Example visual policy:

```python
def choose_action(context):
    return {
        "action": "ACTION1",
        "hypothesis": "Color 2 object moves right.",
        "plan": "Press ACTION1 and verify the visual diff.",
        "expected_state": "NOT_FINISHED",
        "expected_frame_changed": True,
        "expected_changed_cell_count": {"eq": 2},
        "expected_cell": {"x": 1, "y": 1, "color": 2, "layer": 0, "when": "after"},
        "expected_color_delta": {"color": 2, "delta": 0, "layer": 0},
        "expected_object_moved": {"color": 2, "dx": 1, "dy": 0, "layer": 0},
        "expected_visual_predicates": [
            {"op": "cell_changed", "x": 0, "y": 1, "from": 2, "to": 0, "layer": 0},
            {"op": "cell_changed", "x": 1, "y": 1, "from": 0, "to": 2, "layer": 0},
            {"op": "changed_cell_count", "eq": 2}
        ]
    }
```

---

## Monitor Reports

After each executed action, the environment builds a `monitor_report`.

Example:

```json
{
  "phase": "monitor",
  "hypothesis": "Color 2 object moves right.",
  "plan": "Press ACTION1 and verify visual diff.",
  "action": {"action": "ACTION1"},
  "expected": {
    "expected_frame_changed": true,
    "expected_changed_cell_count": {"eq": 2}
  },
  "actual": {
    "state": "NOT_FINISHED",
    "levels_completed": 0,
    "level_delta": 0,
    "won": false,
    "reward": 0.0,
    "frame_changed": true,
    "changed_cell_count": 2,
    "frame_diff": {
      "changed_cell_count": 2,
      "any_change": true
    }
  },
  "visual_checks": [...],
  "expectation_met": true,
  "mismatches": []
}
```

For `action_sequence`, `info` contains:

```json
{
  "action_sequence": [...],
  "action_sequence_executed": [...],
  "actions_executed": 2,
  "monitor_report": {...},
  "monitor_reports": [...],
  "monitor_summary": {...},
  "monitor_summaries": [...],
  "sequence_stopped_reason": "terminal",
  "expectation_met": true
}
```

Sequence stop reasons:

| Reason | Meaning |
|---|---|
| `terminal` | `WIN` or `GAME_OVER` |
| `mismatch` | Scalar or visual expectation failed |
| `invalid` | Invalid Python/action/sequence |
| `max_steps` | Episode step limit reached |
| `max_len` | `max_action_sequence_len` reached |
| `completed` | Whole sequence completed |

---

## Context Compression

Policy context includes full frames for backward compatibility plus compact summaries:

```python
{
    "frames": [...],
    "latest_frame": {...},
    "frame_summaries": [...],
    "latest_frame_summary": {...},
    "last_frame_diff": {...},
    "available_actions": [...],
    "max_action_sequence_len": 3
}
```

Prompt history uses compact monitor summaries instead of dumping full reports.

Recommended settings:

```bash
env.arc_agi_3.context_compression=true \
env.arc_agi_3.max_history_monitor_mismatches=4 \
env.arc_agi_3.max_frame_changed_cells=64 \
env.arc_agi_3.max_prompt_frame_cells=256 \
env.arc_agi_3.include_full_frame_in_prompt=true
```

Keep `include_full_frame_in_prompt=true` initially. If prompts become too large, lower `max_prompt_frame_cells` or later disable full-frame prompt inclusion after validating model behavior.

---

## Config Reference

Default config block in `verl/trainer/config/ppo_trainer.yaml`:

```yaml
env:
  arc_agi_3:
    mode: auto # official|grid|auto
    game_id: null
    games: null
    scorecard_id: null
    data_dir: null
    train_split: train
    val_split: validation
    max_grid_size: 30
    require_think: false
    require_program: true
    program_timeout: 5.0
    program_memory_mb: 512
    reward_correct: 1.0
    reward_wrong: 0.0
    reward_per_level: 0.1
    max_action_sequence_len: 1
    stop_sequence_on_mismatch: true
    stop_sequence_on_invalid: true
    stop_sequence_on_terminal: true
    context_compression: true
    max_history_monitor_mismatches: 4
    max_frame_changed_cells: 64
    max_prompt_frame_cells: 256
    include_full_frame_in_prompt: true
```

---

## Rewards and Termination

Official mode win condition:

```python
won = state == "WIN" or (win_levels > 0 and levels_completed >= win_levels)
```

`done=True` when:

- `won`;
- `state == "GAME_OVER"`;
- `env.max_steps` reached;
- invalid action/program/sequence.

Reward logic:

| Situation | Reward |
|---|---:|
| Win | `reward_correct`, default `1.0` |
| Level progress | `reward_per_level * level_delta` |
| No progress | `reward_wrong`, default `0.0` |
| Invalid policy/action | `reward_wrong`, default `0.0` |

In multi-action mode, step rewards are accumulated, and a winning sequence receives at least `reward_correct`.

---

## Legacy Grid Fallback

For static ARC grid tasks:

```bash
env.arc_agi_3.mode=grid
```

Task format:

```json
{
  "env_kwargs": {
    "mode": "grid",
    "train": [
      {"input": [[1, 0]], "output": [[0, 1]]}
    ],
    "test": [
      {"input": [[2, 0]], "output": [[0, 2]]}
    ]
  }
}
```

The model should define:

```python
def solve(task):
    return output_grid
```

or:

```python
def solve(task):
    return {"outputs": [grid1, grid2]}
```

Grid mode is intended for compatibility and local tests. Use `official` mode for real ARC-AGI-3.

---

## Testing

If the full test environment is available:

```bash
cd /a0/usr/workdir/verl-agent
pytest -q tests/environments/test_arc_agi_3_projection.py tests/environments/test_arc_agi_3_env.py
```

Syntax check:

```bash
python -m compileall -q \
  agent_system/environments/env_package/arc_agi_3 \
  agent_system/environments/prompts/arc_agi_3.py \
  agent_system/environments/env_manager.py \
  tests/environments
```

Smoke scenarios covered by tests/stubs:

- Projection extracts Python policy.
- Grid fallback works.
- Official fake runtime works.
- Reflection loop works.
- Controlled multi-action sequence executes per-step monitors.
- Mismatch stops sequence early.
- Visual frame/cell/color/object predicates pass.
- Visual mismatch stops sequence early.
- Compact context and monitor summaries are present.
- `make_envs()` creates `ArcAGI3EnvironmentManager`.

---

## What to Monitor During Training

Useful fields in rollout logs / `info`:

| Field | Meaning |
|---|---|
| `won` | Whether the game was solved |
| `score` | Reward/score for this step |
| `state` | Current game state |
| `levels_completed` | Progress through game levels |
| `expectation_met` | Whether checked expectations matched reality |
| `sequence_stopped_reason` | Why an action sequence stopped |
| `actions_executed` | How many actions were actually executed |
| `monitor_summary.mismatches` | Compact explanation of wrong assumptions |
| `monitor_report.visual_checks` | Detailed visual predicate results |

Early training problems to watch for:

- Many invalid programs: lower temperature, strengthen prompt, or use SFT warmup.
- Many `mismatch` stops: keep `max_action_sequence_len=1` longer.
- Huge prompts: reduce `history_length` or `max_prompt_frame_cells`.
- Low `expectation_met`: train longer in single-action mode before multi-action.

---

## Recommended Starting Command

```bash
cd /a0/usr/workdir/verl-agent

python -m verl.trainer.main_ppo \
  data.train_files=/a0/usr/workdir/arc_agi_3_data/train.parquet \
  data.val_files=/a0/usr/workdir/arc_agi_3_data/val.parquet \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  env.env_name=arc_agi_3 \
  env.arc_agi_3.mode=official \
  env.arc_agi_3.game_id=ls20 \
  env.max_steps=80 \
  env.history_length=2 \
  env.rollout.n=1 \
  env.arc_agi_3.require_program=true \
  env.arc_agi_3.program_timeout=5.0 \
  env.arc_agi_3.program_memory_mb=512 \
  env.arc_agi_3.max_action_sequence_len=1 \
  env.arc_agi_3.context_compression=true \
  env.arc_agi_3.max_history_monitor_mismatches=4 \
  env.arc_agi_3.max_frame_changed_cells=64 \
  env.arc_agi_3.max_prompt_frame_cells=256 \
  trainer.project_name=arc_agi_3 \
  trainer.experiment_name=arc_agi_3_single_action_smoke \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1
```

After this is stable, try:

```bash
env.arc_agi_3.max_action_sequence_len=2
```

Then:

```bash
env.arc_agi_3.max_action_sequence_len=3
```
