# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# --------------------- ARC-AGI-3 --------------------- #
ARC_AGI_3_TEMPLATE_NO_HIS = """
You are playing an ARC-AGI-3 interactive game environment.

Use this closed-loop method on every turn:

[1. INFERENCE / LLM] Generate a hypothesis about the game rule/state, a plan, and explicit expected outcome(s).
[2. ACTION / Environment] Return one ARC-AGI-3 action or a short controlled `action_sequence`; the environment executes it step by step.
[3. COMPARISON / Monitor] After every executed action, the environment compares your expectation with the actual next Frame.
[4. REFLECTION / LLM] On the next turn, analyze the monitor reports, update working memory, and revise the hypothesis/plan.

ARC-AGI-3 is a dynamic turn-based game API. Your goal is to reach WIN and avoid GAME_OVER while minimizing actions. One action produces one Frame. A Frame contains one or more grids; each grid is a matrix of integer color/object ids. The score is `levels_completed`; the target is `win_levels`.

Task/Game ID: {task_id}
Mode: {mode}

Current observation:
{current_observation}

Available actions:
{available_actions}

Respond with reasoning in <think> </think> tags. Then provide the action payload.

Default fast path, `action_format=json`: provide direct JSON in <action> </action> tags. Do not write Python in this mode. The environment parses the JSON and sends the action/action_sequence directly to ARC runtime without a Python subprocess.

Optional advanced path, `action_format=python`: provide executable Python in <python> </python> tags. Your Python MUST define one of:
1. choose_action(frames, latest_frame) -> action dict or action string
2. choose_action(context) -> action dict or action string
3. solve(context) -> action dict or action string
4. a global variable action / answer with the same format

The returned JSON/action dict SHOULD include these monitorable fields:
- `hypothesis`: your current belief about the rule/dynamics.
- `plan`: why this action follows from the hypothesis.
- `expected_state`: optional expected next state, e.g. "NOT_FINISHED" or "WIN".
- `expected_level_delta`: optional expected change in `levels_completed`.
- `expected_levels_completed`: optional expected absolute score after the action.
- `expected_win`: optional boolean.
- `expected_outcome`: free-text expected visual/game effect.
- `reflection`: optional working-memory note from the previous monitor report.

Visual expectations checked by the monitor:
- `expected_frame_changed`: bool.
- `expected_changed_cell_count`: int or {{"eq": int, "gte": int, "lte": int}}.
- `expected_cell` / `expected_cells`: {{"x": int, "y": int, "color": int, "layer": 0, "when": "after"}}.
- `expected_color_count(s)`: {{"color": int, "count": int, "layer": 0, "when": "after"}}.
- `expected_color_delta(s)`: {{"color": int, "delta": int, "layer": 0}}.
- `expected_object_moved`: {{"color": int, "dx": int, "dy": int, "layer": 0}}.
- `expected_visual_predicates`: safe predicates with op `changed`, `unchanged`, `cell_equals`, `cell_changed`, `color_count`, `color_count_delta`, `changed_cell_count`, `object_moved`, `object_exists`; combinators `all`, `any`, `not` are supported.
Use visual expectations when you can make a concrete prediction about objects, colors, cells, or coordinates.

Action format:
- RESET: start or restart game when state is NOT_PLAYED or GAME_OVER.
- ACTION1: simple input 1 / W / Up.
- ACTION2: simple input 2 / S / Down.
- ACTION3: simple input 3 / A / Left.
- ACTION4: simple input 4 / D / Right.
- ACTION5: simple input 5 / Enter / Spacebar / Delete / perform action.
- ACTION6: click/point action; return {{"action": "ACTION6", "x": int_0_to_63, "y": int_0_to_63}}.
- ACTION7: Undo, if available.

Only choose actions listed in available_actions when possible.

Example JSON response format:
<think>I need to start the game first, then compare the first real frame with my expectations.</think>
<action>{{
  "action": "RESET",
  "hypothesis": "The game has not started yet, so no level logic is visible.",
  "plan": "Reset to obtain the first playable frame.",
  "expected_state": "NOT_FINISHED",
  "expected_level_delta": 0,
  "expected_win": false,
  "expected_outcome": "A first playable frame should appear.",
  "confidence": 1.0
}}</action>

Advanced Python response format, only when action_format=python:
<python>
def choose_action(frames, latest_frame):
    return {{"action": "RESET", "expected_state": "NOT_FINISHED"}}
</python>
"""

ARC_AGI_3_TEMPLATE = """
You are playing an ARC-AGI-3 interactive game environment.

Use this closed-loop method on every turn:

[1. INFERENCE / LLM] Generate a hypothesis about the game rule/state, a plan, and explicit expected outcome(s).
[2. ACTION / Environment] Return one ARC-AGI-3 action or a short controlled `action_sequence`; the environment executes it step by step.
[3. COMPARISON / Monitor] After every executed action, the environment compares your expectation with the actual next Frame.
[4. REFLECTION / LLM] Analyze the monitor reports below, update working memory, and revise the next hypothesis/plan.

Task/Game ID: {task_id}
Mode: {mode}

Current observation:
{current_observation}

Available actions:
{available_actions}

Prior compact monitor summaries, actions, and reflection prompts:
{action_history}

You are now at step {current_step}. Respond with reasoning in <think> </think> tags. In default `action_format=json`, provide direct JSON in <action> </action> tags; do not write Python. In optional `action_format=python`, provide executable Python in <python> </python> tags using choose_action(frames, latest_frame), choose_action(context), solve(context), or global action/answer.
The returned JSON/action dict or each action_sequence item SHOULD include `hypothesis`, `plan`, at least one `expected_*` field, and an updated `reflection` note.
Use `action_sequence` only when the next steps are predictable; the monitor will stop early if any checked expectation fails.
"""

ARC_AGI_3_GRID_TEMPLATE_NO_HIS = """
You are solving a legacy/static ARC grid task in ARC-AGI-3 compatibility mode.

Use this loop:
[1. INFERENCE] infer a transformation hypothesis and implementation plan.
[2. ACTION] write Python solve(task).
[3. COMPARISON] the environment compares your output to hidden test output.
[4. REFLECTION] on retry, update the rule based on the monitor report.

Task ID: {task_id}

Training examples:
{train_examples}

Test input(s):
{test_inputs}

Respond with reasoning in <think> </think> tags. Then provide executable Python in <python> </python> tags. Define solve(task), returning one output grid or {{"outputs": [grid1, grid2, ...]}} for multiple test inputs.
"""

ARC_AGI_3_GRID_TEMPLATE = """
You are solving a legacy/static ARC grid task in ARC-AGI-3 compatibility mode.

Task ID: {task_id}

Training examples:
{train_examples}

Test input(s):
{test_inputs}

Prior monitor reports and reflection prompts:
{action_history}

You are now at attempt {current_step}. Reflect on the monitor report, revise the transformation hypothesis, and provide executable Python in <python> </python> tags. Define solve(task), returning one output grid or {{"outputs": [grid1, grid2, ...]}} for multiple test inputs.
"""
