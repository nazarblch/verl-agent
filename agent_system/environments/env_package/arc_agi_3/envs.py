# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Vectorized ARC-AGI-3 program-execution environment wrapper.

This module follows the public ARC-AGI-3 agent contract from
https://github.com/arcprize/ARC-AGI-3-Agents:

* ARC-AGI-3 is an interactive game environment, not a static grid benchmark.
* Official execution is via ``arc_agi.Arcade().make(game_id)``.
* Each turn returns FrameData-like data: ``frame``, ``state``,
  ``levels_completed``, ``win_levels``, ``available_actions``.
* Actions are ``GameAction`` values: ``RESET`` and ``ACTION1``..``ACTION7``.
  ``ACTION6`` is complex and carries ``x``/``y`` coordinates in ``0..63``.

The model still writes Python, but now that Python is a *policy* that chooses
one interactive action or a controlled short action_sequence from the frame history. A lightweight legacy grid mode is
kept for local tests and datasets that already contain train/test ARC grids.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gym
except Exception:  # pragma: no cover - gym is part of the project deps.
    class _FallbackEnv:
        pass

    class gym:  # type: ignore
        Env = _FallbackEnv


_SIMPLE_ACTIONS = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"}
_COMPLEX_ACTIONS = {"ACTION6"}
_ALL_ACTIONS = _SIMPLE_ACTIONS | _COMPLEX_ACTIONS


def _get_config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _get_arc_config(env_config: Any) -> Any:
    if env_config is None:
        return None
    if isinstance(env_config, dict):
        return env_config.get("arc_agi_3", {})
    return getattr(env_config, "arc_agi_3", None)


def _deepcopy_jsonable(value: Any) -> Any:
    return copy.deepcopy(value)


def _canonical_grid(grid: Any) -> Any:
    return [[int(cell) for cell in row] for row in grid]


def _safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "name"):
        return value.name
    if hasattr(value, "model_dump"):
        return _safe_jsonable(value.model_dump())
    return str(value)


def _is_grid_2d(value: Any) -> bool:
    return isinstance(value, list) and value and all(isinstance(row, list) for row in value) and not any(
        row and isinstance(row[0], list) for row in value if isinstance(row, list)
    )


def _extract_frame_layers(frame: Dict[str, Any]) -> List[Any]:
    """Return best-effort list of 2D grids from an ARC FrameData-like dict."""

    raw = frame.get("frame", []) if isinstance(frame, dict) else []
    if not raw:
        return []
    if _is_grid_2d(raw):
        return [raw]
    if isinstance(raw, list):
        layers = []
        for item in raw:
            if _is_grid_2d(item):
                layers.append(item)
        return layers
    return []


def _grid_shape(grid: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) for row in grid):
        return None
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    return height, width


def _grid_cell(grid: Any, x: int, y: int) -> Any:
    if not isinstance(grid, list) or y < 0 or y >= len(grid):
        raise IndexError(f"y={y} out of bounds")
    row = grid[y]
    if not isinstance(row, list) or x < 0 or x >= len(row):
        raise IndexError(f"x={x} out of bounds")
    return row[x]


def _iter_grid_cells(grid: Any):
    if not isinstance(grid, list):
        return
    for y, row in enumerate(grid):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            yield x, y, value


def _color_counts(grid: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _x, _y, value in _iter_grid_cells(grid):
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _bbox_from_cells(cells: List[Tuple[int, int]]) -> Optional[Dict[str, int]]:
    if not cells:
        return None
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}


def _frame_signature(frame: Dict[str, Any]) -> Dict[str, Any]:
    layers = []
    for grid in _extract_frame_layers(frame):
        shape = _grid_shape(grid) or (0, 0)
        nonzero = [(x, y) for x, y, value in _iter_grid_cells(grid) if value != 0]
        layers.append({
            "height": shape[0],
            "width": shape[1],
            "colors": _color_counts(grid),
            "nonzero_count": len(nonzero),
            "bbox_nonzero": _bbox_from_cells(nonzero),
        })
    return {"num_layers": len(layers), "layers": layers}


def _compute_grid_diff(before_grid: Any, after_grid: Any, max_changed_cells: int = 64) -> Dict[str, Any]:
    before_shape = _grid_shape(before_grid) or (0, 0)
    after_shape = _grid_shape(after_grid) or (0, 0)
    max_h = max(before_shape[0], after_shape[0])
    max_w = max(before_shape[1], after_shape[1])
    changed = []
    sample = []
    for y in range(max_h):
        for x in range(max_w):
            try:
                old = _grid_cell(before_grid, x, y)
            except Exception:
                old = None
            try:
                new = _grid_cell(after_grid, x, y)
            except Exception:
                new = None
            if old != new:
                changed.append((x, y))
                if len(sample) < max_changed_cells:
                    sample.append({"x": x, "y": y, "before": old, "after": new})
    before_counts = _color_counts(before_grid)
    after_counts = _color_counts(after_grid)
    colors = set(before_counts) | set(after_counts)
    color_delta = {c: after_counts.get(c, 0) - before_counts.get(c, 0) for c in sorted(colors)}
    return {
        "shape_before": list(before_shape),
        "shape_after": list(after_shape),
        "shape_changed": before_shape != after_shape,
        "changed_cell_count": len(changed),
        "changed_cells_sample": sample,
        "colors_before": before_counts,
        "colors_after": after_counts,
        "color_delta": color_delta,
        "bbox_changed": _bbox_from_cells(changed),
    }


def _compute_frame_diff(before: Dict[str, Any], after: Dict[str, Any], max_changed_cells: int = 64) -> Dict[str, Any]:
    before_layers = _extract_frame_layers(before)
    after_layers = _extract_frame_layers(after)
    n = max(len(before_layers), len(after_layers))
    layers = []
    total_changed = 0
    for i in range(n):
        bg = before_layers[i] if i < len(before_layers) else []
        ag = after_layers[i] if i < len(after_layers) else []
        diff = _compute_grid_diff(bg, ag, max_changed_cells=max_changed_cells)
        diff["layer"] = i
        layers.append(diff)
        total_changed += int(diff.get("changed_cell_count", 0) or 0)
    return {
        "state_before": before.get("state") if isinstance(before, dict) else None,
        "state_after": after.get("state") if isinstance(after, dict) else None,
        "levels_delta": int(after.get("levels_completed", 0) or 0) - int(before.get("levels_completed", 0) or 0),
        "layers": layers,
        "changed_cell_count": total_changed,
        "any_change": total_changed > 0 or len(before_layers) != len(after_layers),
        "signature_before": _frame_signature(before),
        "signature_after": _frame_signature(after),
    }


def _compare_count(actual: int, expected_spec: Any) -> Tuple[bool, Any]:
    if isinstance(expected_spec, dict):
        if "eq" in expected_spec and actual != int(expected_spec["eq"]):
            return False, expected_spec
        if "gte" in expected_spec and actual < int(expected_spec["gte"]):
            return False, expected_spec
        if "lte" in expected_spec and actual > int(expected_spec["lte"]):
            return False, expected_spec
        return True, expected_spec
    return actual == int(expected_spec), expected_spec


def _layer_grid(frame: Dict[str, Any], layer: int = 0) -> Any:
    layers = _extract_frame_layers(frame)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"layer={layer} out of bounds")
    return layers[layer]


def _connected_components_by_color(grid: Any, background: int = 0, include_background: bool = False) -> List[Dict[str, Any]]:
    seen = set()
    comps = []
    comp_id = 0
    cells_by_pos = {(x, y): value for x, y, value in _iter_grid_cells(grid)}
    for (sx, sy), color in list(cells_by_pos.items()):
        if (sx, sy) in seen:
            continue
        if not include_background and color == background:
            seen.add((sx, sy))
            continue
        stack = [(sx, sy)]
        seen.add((sx, sy))
        cells = []
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in seen or cells_by_pos.get((nx, ny)) != color:
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        comps.append({
            "id": f"c{comp_id}",
            "color": color,
            "area": len(cells),
            "bbox": _bbox_from_cells(cells),
            "centroid": {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys)},
            "cells_sample": [[x, y] for x, y in cells[:32]],
        })
        comp_id += 1
    return comps


def _object_moved_matches(before: Dict[str, Any], after: Dict[str, Any], color: Any, dx: int, dy: int, layer: int = 0) -> bool:
    before_comps = [c for c in _connected_components_by_color(_layer_grid(before, layer)) if c.get("color") == color]
    after_comps = [c for c in _connected_components_by_color(_layer_grid(after, layer)) if c.get("color") == color]
    for b in before_comps:
        for a in after_comps:
            if b.get("area") != a.get("area"):
                continue
            bx, by = b["centroid"]["x"], b["centroid"]["y"]
            ax, ay = a["centroid"]["x"], a["centroid"]["y"]
            if abs((ax - bx) - dx) < 1e-6 and abs((ay - by) - dy) < 1e-6:
                return True
    return False


def _predicate_result(matched: bool, predicate: Dict[str, Any], actual: Any = None, reason: str = "") -> Dict[str, Any]:
    return {"predicate": predicate, "matched": bool(matched), "actual": actual, "reason": reason}


def _evaluate_visual_predicate(predicate: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any], frame_diff: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not isinstance(predicate, dict):
            return _predicate_result(False, predicate, reason="predicate must be a dict")
        if "all" in predicate:
            results = [_evaluate_visual_predicate(p, before, after, frame_diff) for p in predicate.get("all", [])]
            return _predicate_result(all(r["matched"] for r in results), predicate, actual=results)
        if "any" in predicate:
            results = [_evaluate_visual_predicate(p, before, after, frame_diff) for p in predicate.get("any", [])]
            return _predicate_result(any(r["matched"] for r in results), predicate, actual=results)
        if "not" in predicate:
            result = _evaluate_visual_predicate(predicate.get("not"), before, after, frame_diff)
            return _predicate_result(not result["matched"], predicate, actual=result)

        op = str(predicate.get("op", "")).lower()
        layer = int(predicate.get("layer", 0))
        layer_diff = frame_diff.get("layers", [{}])[layer] if layer < len(frame_diff.get("layers", [])) else {}
        if op == "changed":
            return _predicate_result(bool(frame_diff.get("any_change")), predicate, actual=frame_diff.get("changed_cell_count"))
        if op == "unchanged":
            return _predicate_result(not bool(frame_diff.get("any_change")), predicate, actual=frame_diff.get("changed_cell_count"))
        if op == "changed_cell_count":
            ok, spec = _compare_count(int(frame_diff.get("changed_cell_count", 0)), {k: v for k, v in predicate.items() if k in ("eq", "gte", "lte")} or predicate.get("count", 0))
            return _predicate_result(ok, predicate, actual=frame_diff.get("changed_cell_count"), reason=f"expected {spec}")
        if op == "cell_equals":
            when = predicate.get("when", "after")
            grid = _layer_grid(after if when == "after" else before, layer)
            actual = _grid_cell(grid, int(predicate["x"]), int(predicate["y"]))
            return _predicate_result(actual == predicate.get("color"), predicate, actual=actual)
        if op == "cell_changed":
            x, y = int(predicate["x"]), int(predicate["y"])
            old = _grid_cell(_layer_grid(before, layer), x, y)
            new = _grid_cell(_layer_grid(after, layer), x, y)
            ok = old == predicate.get("from") and new == predicate.get("to")
            return _predicate_result(ok, predicate, actual={"from": old, "to": new})
        if op == "color_count":
            when = predicate.get("when", "after")
            counts = _color_counts(_layer_grid(after if when == "after" else before, layer))
            actual = counts.get(str(predicate.get("color")), 0)
            return _predicate_result(actual == int(predicate.get("count", 0)), predicate, actual=actual)
        if op == "color_count_delta":
            actual = int(layer_diff.get("color_delta", {}).get(str(predicate.get("color")), 0))
            return _predicate_result(actual == int(predicate.get("delta", 0)), predicate, actual=actual)
        if op == "object_moved":
            color = predicate.get("color")
            ok = _object_moved_matches(before, after, color, int(predicate.get("dx", 0)), int(predicate.get("dy", 0)), layer)
            return _predicate_result(ok, predicate, actual={"color": color, "dx": predicate.get("dx"), "dy": predicate.get("dy")})
        if op == "object_exists":
            when = predicate.get("when", "after")
            comps = _connected_components_by_color(_layer_grid(after if when == "after" else before, layer))
            comps = [c for c in comps if c.get("color") == predicate.get("color")]
            bbox = predicate.get("bbox")
            if bbox:
                comps = [c for c in comps if c.get("bbox") == bbox]
            return _predicate_result(bool(comps), predicate, actual=comps[:4])
        return _predicate_result(False, predicate, reason=f"unknown visual predicate op {op!r}")
    except Exception as exc:
        return _predicate_result(False, predicate, reason=str(exc))


def _expectation_to_predicates(expected: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    predicates: List[Tuple[str, Dict[str, Any]]] = []
    if "expected_frame_changed" in expected:
        predicates.append(("expected_frame_changed", {"op": "changed" if expected["expected_frame_changed"] else "unchanged"}))
    if "expected_changed_cell_count" in expected:
        spec = expected["expected_changed_cell_count"]
        pred = {"op": "changed_cell_count"}
        if isinstance(spec, dict):
            pred.update(spec)
        else:
            pred["eq"] = spec
        predicates.append(("expected_changed_cell_count", pred))
    cells = []
    for key in ("expected_cell", "expected_coordinate"):
        if key in expected:
            cells.append((key, expected[key]))
    for cell in expected.get("expected_cells", []) if isinstance(expected.get("expected_cells"), list) else []:
        cells.append(("expected_cells", cell))
    for key, cell in cells:
        if isinstance(cell, dict):
            predicates.append((key, {"op": "cell_equals", **cell}))
    for item in expected.get("expected_changed_cells", []) if isinstance(expected.get("expected_changed_cells"), list) else []:
        if isinstance(item, dict):
            pred = {"op": "cell_changed", **item}
            if "before" in pred and "from" not in pred:
                pred["from"] = pred.pop("before")
            if "after" in pred and "to" not in pred:
                pred["to"] = pred.pop("after")
            predicates.append(("expected_changed_cells", pred))
    color_counts = []
    if isinstance(expected.get("expected_color_count"), dict):
        color_counts.append(("expected_color_count", expected["expected_color_count"]))
    color_counts += [("expected_color_counts", x) for x in expected.get("expected_color_counts", []) if isinstance(x, dict)] if isinstance(expected.get("expected_color_counts"), list) else []
    for key, item in color_counts:
        predicates.append((key, {"op": "color_count", **item}))
    color_deltas = []
    if isinstance(expected.get("expected_color_delta"), dict):
        color_deltas.append(("expected_color_delta", expected["expected_color_delta"]))
    color_deltas += [("expected_color_deltas", x) for x in expected.get("expected_color_deltas", []) if isinstance(x, dict)] if isinstance(expected.get("expected_color_deltas"), list) else []
    for key, item in color_deltas:
        predicates.append((key, {"op": "color_count_delta", **item}))
    if isinstance(expected.get("expected_object_moved"), dict):
        predicates.append(("expected_object_moved", {"op": "object_moved", **expected["expected_object_moved"]}))
    if isinstance(expected.get("expected_object_at"), dict):
        predicates.append(("expected_object_at", {"op": "object_exists", **expected["expected_object_at"]}))
    if isinstance(expected.get("expected_objects"), list):
        for item in expected["expected_objects"]:
            if isinstance(item, dict):
                pred = {"op": "object_exists", **item}
                predicates.append(("expected_objects", pred))
    if "expected_visual" in expected:
        visual = expected["expected_visual"]
        if isinstance(visual, list):
            predicates.extend(("expected_visual", p) for p in visual if isinstance(p, dict))
        elif isinstance(visual, dict):
            predicates.append(("expected_visual", visual))
    if isinstance(expected.get("expected_visual_predicates"), list):
        predicates.extend(("expected_visual_predicates", p) for p in expected["expected_visual_predicates"] if isinstance(p, dict))
    return predicates[:64]


def _compact_frame_diff(frame_diff: Dict[str, Any]) -> Dict[str, Any]:
    layers = []
    for layer in frame_diff.get("layers", [])[:3]:
        layers.append({
            "layer": layer.get("layer"),
            "shape_before": layer.get("shape_before"),
            "shape_after": layer.get("shape_after"),
            "changed_cell_count": layer.get("changed_cell_count"),
            "changed_cells_sample": layer.get("changed_cells_sample", [])[:16],
            "color_delta": {k: v for k, v in layer.get("color_delta", {}).items() if v},
            "bbox_changed": layer.get("bbox_changed"),
        })
    return {"changed_cell_count": frame_diff.get("changed_cell_count"), "any_change": frame_diff.get("any_change"), "layers": layers}


def _compact_frame_context(frame: Dict[str, Any], max_cells: int = 256) -> Dict[str, Any]:
    summary = {"state": frame.get("state"), "levels_completed": frame.get("levels_completed"), "win_levels": frame.get("win_levels"), "signature": _frame_signature(frame)}
    layers = _extract_frame_layers(frame)
    small_layers = []
    for grid in layers[:2]:
        shape = _grid_shape(grid) or (0, 0)
        if shape[0] * shape[1] <= max_cells:
            small_layers.append(grid)
    if small_layers:
        summary["frame_sample"] = small_layers
    return summary


def _compact_monitor_report(report: Dict[str, Any], max_mismatches: int = 4) -> Dict[str, Any]:
    actual = report.get("actual", {}) if isinstance(report, dict) else {}
    return {
        "action": report.get("action"),
        "expectation_met": report.get("expectation_met"),
        "actual": {
            "state": actual.get("state"),
            "levels_completed": actual.get("levels_completed"),
            "level_delta": actual.get("level_delta"),
            "won": actual.get("won"),
            "frame_changed": actual.get("frame_changed"),
            "changed_cell_count": actual.get("changed_cell_count"),
        },
        "mismatches": report.get("mismatches", [])[:max_mismatches],
        "reflection": report.get("reflection"),
    }


def _normalize_action_blob(blob: Any) -> Dict[str, Any]:
    if isinstance(blob, str):
        raw = blob.strip()
        if not raw:
            raise ValueError("empty action string")
        blob = {"action": raw}
    if not isinstance(blob, dict):
        raise ValueError("Python policy must return an action dict or action string")

    action = str(blob.get("action", blob.get("name", ""))).upper().strip()
    if action.isdigit():
        action = "RESET" if action == "0" else f"ACTION{action}"
    if action not in _ALL_ACTIONS:
        raise ValueError(f"unsupported ARC-AGI-3 action {action!r}")

    out = dict(blob)
    out["action"] = action
    if action == "ACTION6":
        x = int(out.get("x", 32))
        y = int(out.get("y", 32))
        out["x"] = max(0, min(63, x))
        out["y"] = max(0, min(63, y))
    return out


def _normalize_policy_output(blob: Any) -> Dict[str, Any]:
    """Normalize either a single action or a controlled action_sequence.

    Supported policy outputs:
    * "ACTION1" or {"action": "ACTION1", ...}
    * {"action_sequence": [{"action": "ACTION1", ...}, ...], ...}

    Sequence-level fields such as hypothesis/plan/reflection are copied into
    per-action entries when the entry does not override them, so the monitor can
    still compare every executed step with the policy's intent.
    """

    if isinstance(blob, dict) and "action_sequence" in blob:
        raw_sequence = blob.get("action_sequence")
        if not isinstance(raw_sequence, list) or not raw_sequence:
            raise ValueError("action_sequence must be a non-empty list")
        shared_keys = (
            "hypothesis",
            "plan",
            "reflection",
            "confidence",
            "alternatives_considered",
            "stop_on_mismatch",
            "stop_on_invalid",
            "stop_on_terminal",
        )
        sequence = []
        for step in raw_sequence:
            if isinstance(step, str):
                step_blob = {"action": step}
            elif isinstance(step, dict):
                step_blob = dict(step)
            else:
                raise ValueError("each action_sequence item must be an action dict or string")
            for key in shared_keys:
                if key in blob and key not in step_blob:
                    step_blob[key] = blob[key]
            sequence.append(_normalize_action_blob(step_blob))
        out = {k: v for k, v in blob.items() if k != "action_sequence"}
        out["action_sequence"] = sequence
        return out
    return _normalize_action_blob(blob)


def _policy_action_sequence(policy_output: Dict[str, Any], max_len: int) -> List[Dict[str, Any]]:
    sequence = policy_output.get("action_sequence") if isinstance(policy_output, dict) else None
    if sequence is None:
        sequence = [policy_output]
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("policy produced an empty action sequence")
    max_len = max(1, int(max_len))
    return [_normalize_action_blob(step) for step in sequence[:max_len]]


def _expectation_monitor_report(
    action_blob: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
    won: bool,
    reward: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare scalar and visual expectations with the actual next frame."""

    before_levels = int(before.get("levels_completed", 0) or 0)
    after_levels = int(after.get("levels_completed", 0) or 0)
    frame_diff = _compute_frame_diff(before, after)
    actual = {
        "state": after.get("state"),
        "levels_completed": after_levels,
        "level_delta": after_levels - before_levels,
        "win_levels": after.get("win_levels"),
        "won": bool(won),
        "reward": reward,
        "frame_changed": frame_diff.get("any_change"),
        "changed_cell_count": frame_diff.get("changed_cell_count"),
        "frame_diff": _compact_frame_diff(frame_diff),
        "frame_signature_after": frame_diff.get("signature_after"),
    }
    expected = {
        key: action_blob.get(key)
        for key in (
            "expected_state",
            "expected_level_delta",
            "expected_levels_completed",
            "expected_win",
            "expected_outcome",
            "expected_observation",
            "expected_frame_changed",
            "expected_changed_cell_count",
            "expected_changed_cells",
            "expected_visual",
            "expected_visual_predicates",
            "expected_cell",
            "expected_cells",
            "expected_coordinate",
            "expected_color_count",
            "expected_color_counts",
            "expected_color_delta",
            "expected_color_deltas",
            "expected_colors_after",
            "expected_objects",
            "expected_object_moved",
            "expected_object_at",
        )
        if key in action_blob
    }

    mismatches = []
    checked = 0
    if "expected_state" in expected:
        checked += 1
        if str(expected["expected_state"]) != str(actual["state"]):
            mismatches.append({"field": "state", "expected": expected["expected_state"], "actual": actual["state"]})
    if "expected_level_delta" in expected:
        checked += 1
        try:
            exp_delta = int(expected["expected_level_delta"])
        except Exception:
            exp_delta = expected["expected_level_delta"]
        if exp_delta != actual["level_delta"]:
            mismatches.append({"field": "level_delta", "expected": exp_delta, "actual": actual["level_delta"]})
    if "expected_levels_completed" in expected:
        checked += 1
        try:
            exp_levels = int(expected["expected_levels_completed"])
        except Exception:
            exp_levels = expected["expected_levels_completed"]
        if exp_levels != actual["levels_completed"]:
            mismatches.append({"field": "levels_completed", "expected": exp_levels, "actual": actual["levels_completed"]})
    if "expected_win" in expected:
        checked += 1
        if bool(expected["expected_win"]) != actual["won"]:
            mismatches.append({"field": "won", "expected": bool(expected["expected_win"]), "actual": actual["won"]})
    if "expected_colors_after" in expected and isinstance(expected["expected_colors_after"], dict):
        checked += 1
        layer0 = frame_diff.get("layers", [{}])[0] if frame_diff.get("layers") else {}
        colors_after = layer0.get("colors_after", {})
        exp_colors = {str(k): int(v) for k, v in expected["expected_colors_after"].items()}
        actual_subset = {k: colors_after.get(k, 0) for k in exp_colors}
        if actual_subset != exp_colors:
            mismatches.append({"field": "expected_colors_after", "expected": exp_colors, "actual": actual_subset})

    visual_results = []
    for field, predicate in _expectation_to_predicates(expected):
        checked += 1
        result = _evaluate_visual_predicate(predicate, before, after, frame_diff)
        visual_results.append({"field": field, **result})
        if not result.get("matched"):
            mismatches.append({"field": field, "expected": predicate, "actual": result.get("actual"), "reason": result.get("reason")})

    if error:
        checked += 1
        mismatches.append({"field": "execution_error", "expected": "valid action/program", "actual": error})

    expectation_met = None if checked == 0 else len(mismatches) == 0
    reflection_prompt = (
        "Reflection: compare your hypothesis/plan with the monitor report. "
        "If expectation_met is false, explain the likely mistaken scalar or visual assumption, update working memory, "
        "and choose a revised next action. If true, consolidate what worked."
    )
    return {
        "phase": "monitor",
        "hypothesis": action_blob.get("hypothesis"),
        "plan": action_blob.get("plan"),
        "action": {k: action_blob.get(k) for k in ("action", "x", "y") if k in action_blob},
        "expected": expected,
        "actual": actual,
        "visual_checks": visual_results,
        "expectation_met": expectation_met,
        "mismatches": mismatches,
        "reflection": action_blob.get("reflection"),
        "reflection_prompt": reflection_prompt,
    }

def _grid_monitor_report(
    action: Any,
    prediction_outputs: List[Any],
    expected_outputs: List[Any],
    won: bool,
    reward: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "phase": "monitor",
        "hypothesis": None,
        "plan": "Run the synthesized grid solver and compare produced test output with hidden expected output.",
        "action": "python_grid_solver",
        "expected": {"hidden_outputs_count": len(expected_outputs)},
        "actual": {"prediction": prediction_outputs, "won": bool(won), "reward": reward},
        "expectation_met": bool(won) if not error else False,
        "mismatches": [] if won and not error else [{"field": "grid_output", "expected": "hidden ARC output", "actual": prediction_outputs if not error else error}],
        "reflection": None,
        "reflection_prompt": "Reflection: infer why the generated transformation failed or consolidate the successful rule before the next attempt.",
    }


def _run_python_policy(
    program: str,
    context: Dict[str, Any],
    timeout: float = 5.0,
    memory_mb: int = 512,
) -> Dict[str, Any]:
    """Execute model-written Python and return one action JSON object.

    Accepted user-program contracts:
    * ``choose_action(frames, latest_frame) -> dict|str``
    * ``choose_action(context) -> dict|str``
    * ``solve(context) -> dict|str``
    * global ``action`` / ``answer`` containing a dict or string
    """

    if not isinstance(program, str) or not program.strip():
        raise ValueError("empty Python program")

    harness = "\n".join(
        [
            "import json",
            "import sys",
            "context = json.loads(sys.stdin.read())",
            "frames = context.get('frames', [])",
            "latest_frame = context.get('latest_frame', {})",
            "available_actions = context.get('available_actions', [])",
            "# ---------------- user program starts ----------------",
            program,
            "# ---------------- user program ends ------------------",
            "if 'choose_action' in globals() and callable(globals()['choose_action']):",
            "    try:",
            "        answer = globals()['choose_action'](frames, latest_frame)",
            "    except TypeError:",
            "        answer = globals()['choose_action'](context)",
            "elif 'solve' in globals() and callable(globals()['solve']):",
            "    answer = globals()['solve'](context)",
            "elif 'action' in globals():",
            "    answer = globals()['action']",
            "elif 'answer' in globals():",
            "    answer = globals()['answer']",
            "else:",
            "    raise RuntimeError('Program must define choose_action(...), solve(context), global action, or global answer')",
            "print('__ARC_AGI_3_ACTION__' + json.dumps(answer, separators=(',', ':')))",
            "",
        ]
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as f:
        f.write(harness)
        script_path = f.name

    def _limit_resources():
        try:
            import resource

            address_space = max(64, memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
            cpu_seconds = max(1, int(timeout) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass

    try:
        completed = subprocess.run(
            [sys.executable, "-I", script_path],
            input=json.dumps(context),
            text=True,
            capture_output=True,
            timeout=timeout,
            preexec_fn=_limit_resources if os.name == "posix" else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Python policy timed out after {timeout} seconds") from exc
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"Python policy failed with exit code {completed.returncode}: {stderr[-1000:]}")

    marker = "__ARC_AGI_3_ACTION__"
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith(marker):
            return _normalize_policy_output(json.loads(line[len(marker):]))
    raise RuntimeError("Python policy did not emit an ARC-AGI-3 action")


# ------------------------- legacy static-grid fallback --------------------- #

def _normalize_grid_task(raw_task: Dict[str, Any], default_split: str) -> Dict[str, Any]:
    task = _deepcopy_jsonable(raw_task)

    if "env_kwargs" in task and isinstance(task["env_kwargs"], dict):
        nested = _deepcopy_jsonable(task.pop("env_kwargs"))
        nested.update({k: v for k, v in task.items() if k not in nested})
        task = nested
    elif "extra_info" in task and isinstance(task["extra_info"], dict) and "train" in task["extra_info"]:
        nested = _deepcopy_jsonable(task["extra_info"])
        nested.update({k: v for k, v in task.items() if k not in nested})
        task = nested

    task.setdefault("task_id", task.get("id", "arc_agi_3_grid_task"))
    task.setdefault("data_source", "arc_agi_3")
    task.setdefault("split", default_split)
    task.setdefault("mode", "grid")

    if "train" not in task or "test" not in task:
        raise ValueError("Grid fallback tasks must contain 'train' and 'test'")
    return task


def _extract_expected_outputs(task: Dict[str, Any]) -> List[Any]:
    outputs = []
    for item in task.get("test", []):
        if "output" in item:
            outputs.append(_canonical_grid(item["output"]))
    return outputs


def _public_grid_task(task: Dict[str, Any]) -> Dict[str, Any]:
    public = _deepcopy_jsonable(task)
    public["test"] = [{k: v for k, v in item.items() if k != "output"} for item in public.get("test", [])]
    return public


def _run_legacy_grid_solver(program: str, task: Dict[str, Any], timeout: float, memory_mb: int) -> Any:
    context = _public_grid_task(task)
    blob = _run_python_policy(program, {"task": context, "frames": [], "latest_frame": context, "available_actions": []}, timeout, memory_mb)
    # If policy-style output was used in a grid task, expose the raw dict as a helpful error.
    if "outputs" in blob:
        return blob
    if blob.get("action") in _ALL_ACTIONS:
        raise ValueError("grid fallback expects solve(context) to return a grid or {'outputs': [...]}, not GameAction")
    return blob


def _extract_grid_prediction_outputs(action: Any, task: Dict[str, Any], timeout: float, memory_mb: int) -> List[Any]:
    if isinstance(action, str):
        # For grid fallback, accept programs that define solve(context) and return a grid.
        action = _run_grid_program(action, task, timeout, memory_mb)
    if isinstance(action, dict):
        outputs = action.get("outputs", [])
    else:
        outputs = [action]
    return [_canonical_grid(grid) for grid in outputs]


def _run_grid_program(program: str, task: Dict[str, Any], timeout: float, memory_mb: int) -> Any:
    if not isinstance(program, str) or not program.strip():
        raise ValueError("empty Python program")

    harness = "\n".join(
        [
            "import json",
            "import sys",
            "task = json.loads(sys.stdin.read())",
            "# ---------------- user program starts ----------------",
            program,
            "# ---------------- user program ends ------------------",
            "if 'solve' in globals() and callable(globals()['solve']):",
            "    answer = globals()['solve'](task)",
            "elif 'answer' in globals():",
            "    answer = globals()['answer']",
            "else:",
            "    raise RuntimeError('Program must define solve(task) or global answer')",
            "print('__ARC_AGI_3_ANSWER__' + json.dumps(answer, separators=(',', ':')))",
            "",
        ]
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as f:
        f.write(harness)
        script_path = f.name

    def _limit_resources():
        try:
            import resource

            address_space = max(64, memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
            cpu_seconds = max(1, int(timeout) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass

    try:
        completed = subprocess.run(
            [sys.executable, "-I", script_path],
            input=json.dumps(_public_grid_task(task)),
            text=True,
            capture_output=True,
            timeout=timeout,
            preexec_fn=_limit_resources if os.name == "posix" else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Python grid solver timed out after {timeout} seconds") from exc
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"Python grid solver failed with exit code {completed.returncode}: {stderr[-1000:]}")

    marker = "__ARC_AGI_3_ANSWER__"
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("Python grid solver did not emit an ARC answer")


# ------------------------- official interactive execution ------------------ #

class _OfficialArcRuntime:
    def __init__(self, game_id: str, scorecard_id: Optional[str] = None):
        from arc_agi import Arcade
        from arcengine import GameAction

        self.GameAction = GameAction
        self.game_id = game_id
        self.arcade = Arcade()
        self.env = self.arcade.make(game_id, scorecard_id=scorecard_id)
        self.frames: List[Dict[str, Any]] = []

    def _frame_to_dict(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            raise ValueError("Received None frame data from ARC-AGI-3 environment")
        return {
            "game_id": _safe_jsonable(getattr(raw, "game_id", self.game_id)),
            "frame": _safe_jsonable(getattr(raw, "frame", [])),
            "state": _safe_jsonable(getattr(raw, "state", "UNKNOWN")),
            "levels_completed": int(getattr(raw, "levels_completed", 0) or 0),
            "win_levels": int(getattr(raw, "win_levels", 0) or 0),
            "guid": _safe_jsonable(getattr(raw, "guid", None)),
            "full_reset": bool(getattr(raw, "full_reset", False)),
            "available_actions": _safe_jsonable(getattr(raw, "available_actions", [])),
        }

    def reset(self) -> Dict[str, Any]:
        # ARC-AGI-3 agents start from observation_space and typically choose RESET
        # on NOT_PLAYED/GAME_OVER. We expose that first frame to the policy.
        raw = getattr(self.env, "observation_space", None)
        frame = self._frame_to_dict(raw)
        self.frames = [frame]
        return frame

    def _game_action_from_blob(self, blob: Dict[str, Any]) -> Any:
        GameAction = self.GameAction
        action_name = blob["action"]
        try:
            action = GameAction.from_name(action_name)
        except Exception:
            if action_name == "RESET":
                action = GameAction.RESET
            else:
                action = getattr(GameAction, action_name)
        if action_name == "ACTION6" and hasattr(action, "set_data"):
            action.set_data({"x": int(blob.get("x", 32)), "y": int(blob.get("y", 32))})
        return action

    def step(self, blob: Dict[str, Any]) -> Dict[str, Any]:
        action = self._game_action_from_blob(blob)
        data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
        reasoning = {
            k: blob[k]
            for k in ("thought", "confidence", "alternatives_considered", "reasoning", "hypothesis", "plan", "reflection")
            if k in blob
        }
        raw = self.env.step(action, data=data, reasoning=reasoning or None)
        frame = self._frame_to_dict(raw)
        self.frames.append(frame)
        return frame

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


class ArcAGI3Env(gym.Env):
    """Batched ARC-AGI-3 interactive evaluator with grid fallback."""

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: Any = None,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.env_config = env_config
        self.arc_config = _get_arc_config(env_config)
        self.max_steps = int(_get_config_value(env_config, "max_steps", 80))
        self.program_timeout = float(_get_config_value(self.arc_config, "program_timeout", 5.0))
        self.program_memory_mb = int(_get_config_value(self.arc_config, "program_memory_mb", 512))
        self.reward_correct = float(_get_config_value(self.arc_config, "reward_correct", 1.0))
        self.reward_wrong = float(_get_config_value(self.arc_config, "reward_wrong", 0.0))
        self.reward_per_level = float(_get_config_value(self.arc_config, "reward_per_level", 0.1))
        self.max_action_sequence_len = max(1, int(_get_config_value(self.arc_config, "max_action_sequence_len", 1)))
        self.action_format = str(_get_config_value(self.arc_config, "action_format", "python")).lower()
        self.stop_sequence_on_mismatch = bool(_get_config_value(self.arc_config, "stop_sequence_on_mismatch", True))
        self.stop_sequence_on_invalid = bool(_get_config_value(self.arc_config, "stop_sequence_on_invalid", True))
        self.stop_sequence_on_terminal = bool(_get_config_value(self.arc_config, "stop_sequence_on_terminal", True))
        self.context_compression = bool(_get_config_value(self.arc_config, "context_compression", True))
        self.max_history_monitor_mismatches = int(_get_config_value(self.arc_config, "max_history_monitor_mismatches", 4))
        self.max_frame_changed_cells = int(_get_config_value(self.arc_config, "max_frame_changed_cells", 64))
        self.max_prompt_frame_cells = int(_get_config_value(self.arc_config, "max_prompt_frame_cells", 256))
        self.include_full_frame_in_prompt = bool(_get_config_value(self.arc_config, "include_full_frame_in_prompt", True))
        self.mode = str(_get_config_value(self.arc_config, "mode", "auto")).lower()
        self.default_split = "train" if is_train else _get_config_value(self.arc_config, "val_split", "validation")
        self._dataset = self._load_dataset()
        self._dataset_idx = 0
        self.tasks: List[Dict[str, Any]] = []
        self.runtimes: List[Optional[_OfficialArcRuntime]] = []
        self.steps: List[int] = []
        self.previous_levels: List[int] = []

    def _load_dataset(self) -> List[Dict[str, Any]]:
        data_dir = _get_config_value(self.arc_config, "data_dir", None)
        if not data_dir:
            return []
        split = _get_config_value(self.arc_config, "train_split" if self.is_train else "val_split", "train" if self.is_train else "validation")
        candidates = [
            Path(data_dir) / f"{split}.jsonl",
            Path(data_dir) / f"{split}.json",
            Path(data_dir) / split / "challenges.json",
            Path(data_dir) / f"{split}_challenges.json",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return []
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return [dict(task, task_id=task_id) for task_id, task in payload.items()]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported ARC-AGI-3 dataset format in {os.fspath(path)}")

    def _default_grid_task(self) -> Dict[str, Any]:
        return {
            "task_id": "arc_agi_3_grid_identity_demo",
            "data_source": "arc_agi_3",
            "split": self.default_split,
            "mode": "grid",
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
                {"input": [[2, 2], [0, 0]], "output": [[2, 2], [0, 0]]},
            ],
            "test": [{"input": [[3, 0], [0, 3]], "output": [[3, 0], [0, 3]]}],
        }

    def _default_official_task(self) -> Dict[str, Any]:
        game_id = _get_config_value(self.arc_config, "game_id", None)
        games = _get_config_value(self.arc_config, "games", None)
        if game_id is None and games:
            game_id = list(games)[0]
        return {
            "task_id": str(game_id or "arc_agi_3_game"),
            "game_id": game_id,
            "data_source": "arc_agi_3",
            "split": self.default_split,
            "mode": "official",
        }

    def _next_dataset_tasks(self, n: int) -> List[Dict[str, Any]]:
        if self._dataset:
            tasks = []
            for _ in range(n):
                tasks.append(self._dataset[self._dataset_idx % len(self._dataset)])
                self._dataset_idx += 1
            return tasks
        if self.mode == "grid":
            return [self._default_grid_task() for _ in range(n)]
        return [self._default_official_task() for _ in range(n)]

    def _expand_kwargs(self, kwargs: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        if kwargs is None:
            return self._next_dataset_tasks(self.batch_size)
        if len(kwargs) == self.env_num and self.group_n > 1:
            expanded = []
            for item in kwargs:
                expanded.extend([item] * self.group_n)
            return expanded
        return list(kwargs)

    def _task_is_grid(self, task: Dict[str, Any]) -> bool:
        return self.mode == "grid" or "train" in task or "test" in task

    def _normalize_official_task(self, raw_task: Dict[str, Any]) -> Dict[str, Any]:
        task = _deepcopy_jsonable(raw_task)
        if "env_kwargs" in task and isinstance(task["env_kwargs"], dict):
            nested = _deepcopy_jsonable(task.pop("env_kwargs"))
            nested.update({k: v for k, v in task.items() if k not in nested})
            task = nested
        game_id = task.get("game_id") or task.get("task_id") or _get_config_value(self.arc_config, "game_id", None)
        if not game_id:
            raise ValueError("Official ARC-AGI-3 mode requires task/game_id or env.arc_agi_3.game_id")
        task.setdefault("task_id", str(game_id))
        task["game_id"] = str(game_id)
        task.setdefault("data_source", "arc_agi_3")
        task.setdefault("split", self.default_split)
        task.setdefault("mode", "official")
        return task

    def reset(self, kwargs: List[Dict[str, Any]] | None = None):
        raw_tasks = self._expand_kwargs(kwargs)
        if len(raw_tasks) > self.batch_size:
            raise ValueError(f"Got {len(raw_tasks)} ARC-AGI-3 tasks, but total_envs={self.batch_size}")

        pad_n = self.batch_size - len(raw_tasks)
        padded = raw_tasks + self._next_dataset_tasks(pad_n)
        valid_mask = [True] * len(raw_tasks) + [False] * pad_n

        self.tasks = []
        self.runtimes = []
        self.steps = []
        self.previous_levels = []
        obs_all = []
        infos_all = []

        for raw in padded:
            if self._task_is_grid(raw):
                task = _normalize_grid_task(raw, self.default_split)
                runtime = None
                obs = _deepcopy_jsonable(task)
                info = {"task_id": task.get("task_id"), "data_source": task.get("data_source", "arc_agi_3"), "split": task.get("split", self.default_split), "mode": "grid", "won": False}
                prev_levels = 0
            else:
                task = self._normalize_official_task(raw)
                scorecard_id = _get_config_value(self.arc_config, "scorecard_id", None)
                runtime = _OfficialArcRuntime(task["game_id"], scorecard_id=scorecard_id)
                frame = runtime.reset()
                obs = {"mode": "official", "task_id": task["task_id"], "game_id": task["game_id"], "frames": runtime.frames, "latest_frame": frame, "latest_frame_summary": _compact_frame_context(frame, self.max_prompt_frame_cells)}
                info = {"task_id": task["task_id"], "game_id": task["game_id"], "data_source": task.get("data_source", "arc_agi_3"), "split": task.get("split", self.default_split), "mode": "official", "won": False, "levels_completed": frame.get("levels_completed", 0), "win_levels": frame.get("win_levels", 0), "state": frame.get("state")}
                prev_levels = int(frame.get("levels_completed", 0) or 0)
            self.tasks.append(task)
            self.runtimes.append(runtime)
            self.steps.append(0)
            self.previous_levels.append(prev_levels)
            obs_all.append(obs)
            infos_all.append(info)

        obs_list = [o for o, keep in zip(obs_all, valid_mask) if keep]
        info_list = [i for i, keep in zip(infos_all, valid_mask) if keep]
        return obs_list, info_list

    def _step_grid(self, idx: int, task: Dict[str, Any], action: Any):
        expected_outputs = _extract_expected_outputs(task)
        error = None
        try:
            prediction_outputs = _extract_grid_prediction_outputs(action, task, self.program_timeout, self.program_memory_mb)
        except Exception as exc:
            prediction_outputs = []
            error = str(exc)
        won = bool(expected_outputs) and prediction_outputs == expected_outputs
        done = bool(won or self.steps[idx] >= self.max_steps)
        reward = self.reward_correct if won else self.reward_wrong
        feedback = "Correct. Python grid solver output matches hidden output." if won else "Incorrect. Return the correct ARC grid from solve(task)."
        if error:
            feedback = f"Invalid ARC-AGI-3 Python grid solver: {error}"
        monitor_report = _grid_monitor_report(action, prediction_outputs, expected_outputs, won, reward, error)
        next_obs = _deepcopy_jsonable(task)
        next_obs.update({"feedback": feedback, "monitor_report": monitor_report, "reflection_prompt": monitor_report["reflection_prompt"], "previous_prediction": prediction_outputs, "step_count": self.steps[idx]})
        info = {"task_id": task.get("task_id"), "data_source": task.get("data_source", "arc_agi_3"), "split": task.get("split", self.default_split), "mode": "grid", "won": won, "score": reward, "expected": expected_outputs, "prediction": prediction_outputs, "monitor_report": monitor_report, "step_count": self.steps[idx]}
        if error:
            info["error"] = error
        return next_obs, reward, done, info

    def _step_official(self, idx: int, task: Dict[str, Any], runtime: _OfficialArcRuntime, action_payload: Any):
        """Execute a controlled policy action_sequence with per-step monitoring.

        The LLM still runs once and returns either one action or an
        ``action_sequence``. The environment executes at most
        ``max_action_sequence_len`` actions, compares each expected outcome with
        the actual next frame, and stops early on mismatch/invalid/terminal when
        configured. This keeps reflection and credit assignment while reducing
        LLM calls on predictable multi-step plans.
        """

        error = None
        policy_output: Dict[str, Any] = {}
        action_sequence: List[Dict[str, Any]] = []
        executed_actions: List[Dict[str, Any]] = []
        monitor_reports: List[Dict[str, Any]] = []
        total_reward = 0.0
        sequence_stopped_reason = "max_len"
        frame = runtime.frames[-1] if runtime.frames else {}
        state = str(frame.get("state", "UNKNOWN"))
        levels = int(frame.get("levels_completed", 0) or 0)
        win_levels = int(frame.get("win_levels", 0) or 0)
        won = state == "WIN" or (win_levels > 0 and levels >= win_levels)

        try:
            before_frame = runtime.frames[-1]
            context = {
                "phase": "inference",
                "instruction": "Generate a hypothesis, plan, expected outcome, and one action or controlled action_sequence.",
                "game_id": task["game_id"],
                "frames": runtime.frames,
                "latest_frame": before_frame,
                "available_actions": before_frame.get("available_actions", []),
                "max_action_sequence_len": self.max_action_sequence_len,
                "frame_summaries": [_compact_frame_context(f, self.max_prompt_frame_cells) for f in runtime.frames],
                "latest_frame_summary": _compact_frame_context(before_frame, self.max_prompt_frame_cells),
                "last_frame_diff": _compact_frame_diff(_compute_frame_diff(runtime.frames[-2], runtime.frames[-1], self.max_frame_changed_cells)) if len(runtime.frames) >= 2 else None,
            }
            if self.action_format == "json":
                policy_output = _normalize_policy_output(action_payload)
            else:
                policy_output = _run_python_policy(action_payload, context, self.program_timeout, self.program_memory_mb)
            remaining_actions = max(0, self.max_steps - (self.steps[idx] - 1))
            allowed_len = min(self.max_action_sequence_len, max(1, remaining_actions))
            action_sequence = _policy_action_sequence(policy_output, allowed_len)

            for seq_idx, action_blob in enumerate(action_sequence):
                before_frame = runtime.frames[-1]
                frame = runtime.step(action_blob)
                executed_actions.append(action_blob)

                state = str(frame.get("state", "UNKNOWN"))
                levels = int(frame.get("levels_completed", 0) or 0)
                win_levels = int(frame.get("win_levels", 0) or 0)
                level_delta = max(0, levels - self.previous_levels[idx])
                self.previous_levels[idx] = levels
                won = state == "WIN" or (win_levels > 0 and levels >= win_levels)
                step_count_after_action = self.steps[idx] + seq_idx
                terminal_by_steps = step_count_after_action >= self.max_steps
                done_after_action = bool(won or state == "GAME_OVER" or terminal_by_steps)
                step_reward = self.reward_correct if won else (self.reward_per_level * level_delta if level_delta else self.reward_wrong)
                total_reward += step_reward

                report = _expectation_monitor_report(action_blob, before_frame, frame, won, step_reward, None)
                report["sequence_index"] = seq_idx
                report["step_count"] = step_count_after_action
                monitor_reports.append(report)

                if done_after_action:
                    sequence_stopped_reason = "terminal" if (won or state == "GAME_OVER") else "max_steps"
                    break
                if self.stop_sequence_on_mismatch and report.get("expectation_met") is False:
                    sequence_stopped_reason = "mismatch"
                    break
            else:
                sequence_stopped_reason = "completed" if len(action_sequence) < self.max_action_sequence_len else "max_len"

            if executed_actions:
                self.steps[idx] += len(executed_actions) - 1
            elif not action_sequence:
                sequence_stopped_reason = "empty_sequence"
        except Exception as exc:
            action_blob = {}
            before_frame = runtime.frames[-1] if runtime.frames else {}
            frame = before_frame
            error = str(exc)
            report = _expectation_monitor_report(action_blob, before_frame, frame, won, self.reward_wrong, error)
            report["sequence_index"] = 0
            report["step_count"] = self.steps[idx]
            monitor_reports.append(report)
            total_reward = self.reward_wrong
            sequence_stopped_reason = "invalid" if self.stop_sequence_on_invalid else "error"

        state = str(frame.get("state", "UNKNOWN"))
        levels = int(frame.get("levels_completed", 0) or 0)
        win_levels = int(frame.get("win_levels", 0) or 0)
        won = state == "WIN" or (win_levels > 0 and levels >= win_levels)
        done = bool(won or state == "GAME_OVER" or self.steps[idx] >= self.max_steps or sequence_stopped_reason == "invalid")
        reward = float(total_reward)
        if won and reward < self.reward_correct:
            reward = self.reward_correct

        monitor_report = monitor_reports[-1] if monitor_reports else _expectation_monitor_report({}, frame, frame, won, reward, error)
        monitor_summaries = [_compact_monitor_report(r, self.max_history_monitor_mismatches) for r in monitor_reports]
        feedback_payload = {
            "sequence_stopped_reason": sequence_stopped_reason,
            "actions_executed": len(executed_actions),
            "monitor_report": monitor_report,
            "monitor_reports": monitor_reports,
            "monitor_summary": _compact_monitor_report(monitor_report, self.max_history_monitor_mismatches),
            "monitor_summaries": monitor_summaries,
        }
        feedback = f"Monitor report: {json.dumps(feedback_payload, ensure_ascii=False, separators=(',', ':'))}"
        if error:
            feedback = f"Invalid ARC-AGI-3 action/action_sequence: {error}. " + feedback
        next_obs = {
            "mode": "official",
            "task_id": task["task_id"],
            "game_id": task["game_id"],
            "frames": runtime.frames,
            "latest_frame": frame,
            "latest_frame_summary": _compact_frame_context(frame, self.max_prompt_frame_cells),
            "feedback": feedback,
            "monitor_report": feedback_payload,
            "monitor_summary": feedback_payload["monitor_summary"],
            "reflection_prompt": monitor_report.get("reflection_prompt", ""),
            "step_count": self.steps[idx],
        }
        info = {
            "task_id": task["task_id"],
            "game_id": task["game_id"],
            "data_source": task.get("data_source", "arc_agi_3"),
            "split": task.get("split", self.default_split),
            "mode": "official",
            "won": won,
            "score": reward,
            "state": state,
            "levels_completed": levels,
            "win_levels": win_levels,
            "available_actions": frame.get("available_actions", []),
            "action": executed_actions[-1] if executed_actions else policy_output,
            "action_sequence": action_sequence,
            "action_sequence_executed": executed_actions,
            "actions_executed": len(executed_actions),
            "monitor_report": monitor_report,
            "monitor_reports": monitor_reports,
            "monitor_summary": feedback_payload["monitor_summary"],
            "monitor_summaries": monitor_summaries,
            "latest_frame_summary": _compact_frame_context(frame, self.max_prompt_frame_cells),
            "sequence_stopped_reason": sequence_stopped_reason,
            "expectation_met": monitor_report.get("expectation_met"),
            "step_count": self.steps[idx],
        }
        if error:
            info["error"] = error
        return next_obs, reward, done, info

    def step(self, actions: List[Any]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} ARC-AGI-3 actions, but total_envs={self.batch_size}")
        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        for idx, (task, runtime, action) in enumerate(zip(self.tasks, self.runtimes, padded_actions)):
            self.steps[idx] += 1
            if task.get("mode") == "grid":
                out = self._step_grid(idx, task, action)
            else:
                if runtime is None:
                    raise RuntimeError("Official ARC-AGI-3 runtime is not initialized")
                out = self._step_official(idx, task, runtime, action)
            obs, reward, done, info = out
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        return obs_list, reward_list, done_list, info_list

    def close(self):
        for runtime in getattr(self, "runtimes", []):
            if runtime is not None:
                runtime.close()


def build_arc_agi_3_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Any = None,
    resources_per_worker: Any = None,
):
    del resources_per_worker
    return ArcAGI3Env(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train, env_config=env_config)
