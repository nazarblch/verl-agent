from agent_system.environments.env_package.arc_agi_3 import arc_agi_3_projection


def test_arc_agi_3_projection_valid_python_tag():
    actions, valids = arc_agi_3_projection(["<think>x</think><python>def solve(task):\n    return [[1]]</python>"], require_think=True)
    assert "def solve" in actions[0]
    assert valids == [1]


def test_arc_agi_3_projection_valid_markdown_code():
    actions, valids = arc_agi_3_projection(["```python\ndef solve(task):\n    return [[1]]\n```"])
    assert "def solve" in actions[0]
    assert valids == [1]


def test_arc_agi_3_projection_requires_program_by_default():
    actions, valids = arc_agi_3_projection(["<answer>[[1]]</answer>"])
    assert actions == [""]
    assert valids == [0]


def test_arc_agi_3_projection_direct_answer_debug_mode():
    actions, valids = arc_agi_3_projection(["<answer>[[0,1],[1,0]]</answer>"], require_program=False)
    assert actions == [[[0, 1], [1, 0]]]
    assert valids == [1]


def test_arc_agi_3_projection_requires_think():
    actions, valids = arc_agi_3_projection(["<python>def solve(task):\n    return [[1]]</python>"], require_think=True)
    assert "def solve" in actions[0]
    assert valids == [0]


def test_arc_agi_3_projection_direct_json_action():
    actions, valids = arc_agi_3_projection(['<think>x</think><action>{"action":"RESET","expected_state":"NOT_FINISHED"}</action>'], require_think=True, action_format="json")
    assert actions == [{"action": "RESET", "expected_state": "NOT_FINISHED"}]
    assert valids == [1]


def test_arc_agi_3_projection_direct_json_sequence():
    text = '<action>{"action_sequence":[{"action":"RESET"},{"action":"ACTION1"}],"stop_on_mismatch":true}</action>'
    actions, valids = arc_agi_3_projection([text], action_format="json")
    assert actions[0]["action_sequence"][0]["action"] == "RESET"
    assert actions[0]["action_sequence"][1]["action"] == "ACTION1"
    assert valids == [1]


def test_arc_agi_3_projection_json_rejects_python_only():
    actions, valids = arc_agi_3_projection(["<python>def choose_action(context): return {'action':'RESET'}</python>"], action_format="json")
    assert actions == [""]
    assert valids == [0]
