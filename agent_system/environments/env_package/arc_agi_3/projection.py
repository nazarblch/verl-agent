# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Projection utilities for ARC-AGI-3 program-synthesis actions."""

from __future__ import annotations

import json
import re
from typing import Any, List, Tuple

_CODE_BLOCK_RE = re.compile(r"<(?:python|code)>(.*?)</(?:python|code)>", re.IGNORECASE | re.DOTALL)
_MARKDOWN_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ACTION_BLOCK_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def _fallback_action() -> str:
    return ""


def _is_valid_grid(value: Any, max_grid_size: int = 30) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if len(value) > max_grid_size:
        return False
    width = None
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        if len(row) > max_grid_size:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool):
                return False
            if cell < 0 or cell > 9:
                return False
    return True


def _is_valid_outputs(value: Any, max_grid_size: int = 30) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_valid_grid(grid, max_grid_size) for grid in value)


def _parse_answer_payload(payload: str, max_grid_size: int) -> tuple[Any, bool]:
    parsed = json.loads(payload)
    if isinstance(parsed, dict):
        outputs = parsed.get("outputs")
        if not _is_valid_outputs(outputs, max_grid_size):
            return _fallback_action(), False
        return {"outputs": outputs}, True
    if _is_valid_grid(parsed, max_grid_size):
        return parsed, True
    if _is_valid_outputs(parsed, max_grid_size):
        return {"outputs": parsed}, True
    return _fallback_action(), False


def arc_agi_3_projection(
    actions: List[str],
    require_think: bool = False,
    max_grid_size: int = 30,
    require_program: bool = True,
    action_format: str = "python",
) -> Tuple[List[Any], List[int]]:
    """Extract ARC-AGI-3 actions from LLM responses.

    ``action_format="json"`` extracts direct JSON from ``<action>...</action>``
    and returns the parsed action/action_sequence without requiring Python.

    ``action_format="python"`` extracts Python solver/policy programs.

    Preferred Python ARC-AGI-3 action format::

        <think>...</think>
        <python>
        def solve(task):
            ...
            return [[0, 1], [1, 0]]
        </python>

    The environment executes the returned program with a public task dict that
    includes train examples and test inputs, but not hidden test outputs.

    For backward compatibility/debugging, ``require_program=False`` still allows
    direct ``<answer>...</answer>`` grid predictions.
    """

    results: List[Any] = []
    valids: List[int] = []
    fmt = str(action_format or "python").lower()

    for action in actions:
        action = action or ""
        valid = 1
        if require_think and _THINK_BLOCK_RE.search(action) is None:
            valid = 0

        if fmt == "json":
            action_matches = _ACTION_BLOCK_RE.findall(action)
            if len(action_matches) != 1:
                results.append(_fallback_action())
                valids.append(0)
                continue
            try:
                parsed = json.loads(action_matches[0].strip())
            except Exception:
                results.append(_fallback_action())
                valids.append(0)
                continue
            if not isinstance(parsed, (dict, list, str)):
                valid = 0
            results.append(parsed)
            valids.append(int(valid and isinstance(parsed, (dict, list, str))))
            continue

        code_matches = _CODE_BLOCK_RE.findall(action)
        if not code_matches:
            code_matches = _MARKDOWN_CODE_RE.findall(action)

        if code_matches:
            if len(code_matches) != 1:
                valid = 0
            code = code_matches[0].strip()
            if not code:
                valid = 0
            results.append(code)
            valids.append(int(valid and bool(code)))
            continue

        if require_program:
            results.append(_fallback_action())
            valids.append(0)
            continue

        answer_match = _ANSWER_BLOCK_RE.search(action)
        if answer_match is None:
            results.append(_fallback_action())
            valids.append(0)
            continue
        if len(_ANSWER_BLOCK_RE.findall(action)) != 1:
            valid = 0
        try:
            parsed, parsed_valid = _parse_answer_payload(answer_match.group(1).strip(), max_grid_size)
        except Exception:
            parsed, parsed_valid = _fallback_action(), False
        results.append(parsed)
        valids.append(int(valid and parsed_valid))

    return results, valids
