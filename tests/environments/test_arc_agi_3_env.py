from omegaconf import OmegaConf

from agent_system.environments.env_manager import ArcAGI3EnvironmentManager, make_envs
from agent_system.environments.env_package.arc_agi_3 import build_arc_agi_3_envs, arc_agi_3_projection



def _install_fake_arc_runtime(monkeypatch):
    import sys
    import types
    from enum import Enum

    class FakeActionData:
        def __init__(self):
            self.payload = {}

        def model_dump(self):
            return dict(self.payload)

    class FakeGameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        @classmethod
        def from_name(cls, name):
            return cls[name.upper()]

        def is_complex(self):
            return self is FakeGameAction.ACTION6

        def is_simple(self):
            return not self.is_complex()

        @property
        def action_data(self):
            if not hasattr(self, "_fake_action_data"):
                self._fake_action_data = FakeActionData()
            return self._fake_action_data

        def set_data(self, data):
            self.action_data.payload = dict(data)

    class FakeFrame:
        def __init__(self, state="NOT_PLAYED", levels_completed=0, win_levels=1):
            self.game_id = "ls20"
            self.frame = [[[0, 1], [2, 3]]]
            self.state = types.SimpleNamespace(name=state)
            self.levels_completed = levels_completed
            self.win_levels = win_levels
            self.guid = "fake-guid"
            self.full_reset = False
            self.available_actions = [FakeGameAction.RESET, FakeGameAction.ACTION1, FakeGameAction.ACTION6]

    class FakeEnv:
        def __init__(self):
            self.observation_space = FakeFrame("NOT_PLAYED", 0, 1)

        def step(self, action, data=None, reasoning=None):
            if action is FakeGameAction.RESET:
                return FakeFrame("NOT_FINISHED", 0, 1)
            return FakeFrame("WIN", 1, 1)

    class FakeArcade:
        def make(self, game_id, scorecard_id=None):
            return FakeEnv()

    arc_agi = types.ModuleType("arc_agi")
    arc_agi.Arcade = FakeArcade
    arcengine = types.ModuleType("arcengine")
    arcengine.GameAction = FakeGameAction
    monkeypatch.setitem(sys.modules, "arc_agi", arc_agi)
    monkeypatch.setitem(sys.modules, "arcengine", arcengine)



def _install_fake_arc_runtime_with_visual_change(monkeypatch):
    import sys
    import types
    from enum import Enum

    class FakeActionData:
        def __init__(self):
            self.payload = {}

        def model_dump(self):
            return dict(self.payload)

    class FakeGameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        @classmethod
        def from_name(cls, name):
            return cls[name.upper()]

        @property
        def action_data(self):
            if not hasattr(self, "_fake_action_data"):
                self._fake_action_data = FakeActionData()
            return self._fake_action_data

        def set_data(self, data):
            self.action_data.payload = dict(data)

    class FakeFrame:
        def __init__(self, state="NOT_PLAYED", levels_completed=0, win_levels=1, grid=None):
            self.game_id = "ls20"
            self.frame = [grid if grid is not None else [[0, 0, 0], [2, 0, 0], [0, 0, 0]]]
            self.state = types.SimpleNamespace(name=state)
            self.levels_completed = levels_completed
            self.win_levels = win_levels
            self.guid = "fake-guid"
            self.full_reset = False
            self.available_actions = [FakeGameAction.RESET, FakeGameAction.ACTION1, FakeGameAction.ACTION6]

    class FakeEnv:
        def __init__(self):
            self.observation_space = FakeFrame("NOT_PLAYED", 0, 1, [[0, 0, 0], [2, 0, 0], [0, 0, 0]])

        def step(self, action, data=None, reasoning=None):
            if action is FakeGameAction.RESET:
                return FakeFrame("NOT_FINISHED", 0, 1, [[0, 0, 0], [2, 0, 0], [0, 0, 0]])
            if action is FakeGameAction.ACTION1:
                return FakeFrame("NOT_FINISHED", 0, 1, [[0, 0, 0], [0, 2, 0], [0, 0, 0]])
            return FakeFrame("WIN", 1, 1, [[0, 0, 0], [0, 2, 0], [0, 0, 0]])

    class FakeArcade:
        def make(self, game_id, scorecard_id=None):
            return FakeEnv()

    arc_agi = types.ModuleType("arc_agi")
    arc_agi.Arcade = FakeArcade
    arcengine = types.ModuleType("arcengine")
    arcengine.GameAction = FakeGameAction
    monkeypatch.setitem(sys.modules, "arc_agi", arc_agi)
    monkeypatch.setitem(sys.modules, "arcengine", arcengine)


def _grid_task(task_id="copy"):
    return {
        "task_id": task_id,
        "data_source": "arc_agi_3",
        "split": "train",
        "mode": "grid",
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
            {"input": [[2, 2], [0, 0]], "output": [[2, 2], [0, 0]]},
        ],
        "test": [{"input": [[3, 0], [0, 3]], "output": [[3, 0], [0, 3]]}],
    }


def _official_task(game_id="ls20"):
    return {"task_id": game_id, "game_id": game_id, "data_source": "arc_agi_3", "mode": "official"}


def _config(train_batch_size=1, val_batch_size=1, group_n=1, mode="grid", max_action_sequence_len=1):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": train_batch_size, "val_batch_size": val_batch_size},
            "env": {
                "env_name": "arc_agi_3",
                "seed": 0,
                "max_steps": 2,
                "history_length": 0,
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
                "rollout": {"n": group_n},
                "arc_agi_3": {
                    "mode": mode,
                    "game_id": "ls20",
                    "games": None,
                    "scorecard_id": None,
                    "data_dir": None,
                    "train_split": "train",
                    "val_split": "validation",
                    "max_grid_size": 30,
                    "require_think": False,
                    "require_program": True,
                    "program_timeout": 5.0,
                    "program_memory_mb": 512,
                    "reward_correct": 1.0,
                    "reward_wrong": 0.0,
                    "reward_per_level": 0.1,
                    "max_action_sequence_len": max_action_sequence_len,
                    "stop_sequence_on_mismatch": True,
                    "stop_sequence_on_invalid": True,
                    "stop_sequence_on_terminal": True,
                    "context_compression": True,
                    "max_history_monitor_mismatches": 4,
                    "max_frame_changed_cells": 64,
                    "max_prompt_frame_cells": 256,
                    "include_full_frame_in_prompt": True,
                },
            },
        }
    )


def test_arc_agi_3_grid_env_reset_and_correct_step():
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=_config().env)
    obs, infos = env.reset([_grid_task()])
    assert len(obs) == 1
    assert infos[0]["task_id"] == "copy"

    program = "def solve(task):\n    return [row[:] for row in task['test'][0]['input']]"
    obs, rewards, dones, infos = env.step([program])
    assert rewards == [1.0]
    assert dones == [True]
    assert infos[0]["won"] is True
    assert infos[0]["prediction"] == [[[3, 0], [0, 3]]]


def test_arc_agi_3_grid_env_wrong_step():
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=_config().env)
    env.reset([_grid_task()])
    _obs, rewards, dones, infos = env.step(["def solve(task):\n    return [[0]]"])
    assert rewards == [0.0]
    assert dones == [False]  # max_steps=2, so one retry remains
    assert infos[0]["won"] is False


def test_arc_agi_3_grid_env_group_n_repeats_tasks():
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=2, is_train=True, env_config=_config(group_n=2).env)
    obs, infos = env.reset([_grid_task("grouped")])
    assert len(obs) == 2
    assert [info["task_id"] for info in infos] == ["grouped", "grouped"]


def test_arc_agi_3_manager_builds_grid_prompt_and_action_validity():
    cfg = _config()
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=cfg.env)
    manager = ArcAGI3EnvironmentManager(env, arc_agi_3_projection, cfg)
    observations, _infos = manager.reset([_grid_task()])
    assert "legacy/static ARC grid" in observations["text"][0]
    next_observations, rewards, dones, infos = manager.step(["<python>def solve(task):\n    return [row[:] for row in task['test'][0]['input']]</python>"])
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
    assert infos[0]["is_action_valid"].item() == 1
    assert next_observations["image"] is None


def test_arc_agi_3_official_prompt_shape_with_fake_runtime(monkeypatch):
    _install_fake_arc_runtime(monkeypatch)
    cfg = _config(mode="official")
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=cfg.env)
    manager = ArcAGI3EnvironmentManager(env, arc_agi_3_projection, cfg)
    observations, infos = manager.reset([_official_task("ls20")])
    text = observations["text"][0]
    assert "interactive game environment" in text
    assert "[1. INFERENCE / LLM]" in text
    assert "[3. COMPARISON / Monitor]" in text
    assert "[4. REFLECTION / LLM]" in text
    assert "choose_action(frames, latest_frame)" in text
    assert "ACTION6" in text
    assert infos[0]["mode"] == "official"

    program = """def choose_action(frames, latest_frame):
    return {
        'action': 'RESET',
        'hypothesis': 'The game is not started.',
        'plan': 'Reset to get a playable frame.',
        'expected_state': 'NOT_FINISHED',
        'expected_level_delta': 0,
        'expected_win': False,
        'expected_outcome': 'A playable frame appears.',
        'reflection': 'Start by resetting.'
    }
"""
    next_observations, rewards, dones, infos = manager.step([f"<python>{program}</python>"])
    assert infos[0]["action"]["action"] == "RESET"
    assert infos[0]["state"] in {"NOT_FINISHED", "WIN"}
    assert "monitor_report" in infos[0]
    assert infos[0]["monitor_report"]["hypothesis"] == "The game is not started."
    assert infos[0]["monitor_report"]["expected"]["expected_state"] == "NOT_FINISHED"
    assert "Prior monitor reports" in next_observations["text"][0]


def test_make_envs_arc_agi_3_smoke():
    cfg = _config(train_batch_size=1, val_batch_size=1)
    train_envs, val_envs = make_envs(cfg)
    assert isinstance(train_envs, ArcAGI3EnvironmentManager)
    assert isinstance(val_envs, ArcAGI3EnvironmentManager)



def test_arc_agi_3_official_controlled_multi_action_sequence(monkeypatch):
    _install_fake_arc_runtime(monkeypatch)
    cfg = _config(mode="official", max_action_sequence_len=3)
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=cfg.env)
    manager = ArcAGI3EnvironmentManager(env, arc_agi_3_projection, cfg)
    observations, _infos = manager.reset([_official_task("ls20")])
    assert "action_sequence" in observations["text"][0]

    program = """def choose_action(frames, latest_frame):
    return {
        'hypothesis': 'Reset then use ACTION1 to finish.',
        'plan': 'Execute two predictable actions with per-step expectations.',
        'action_sequence': [
            {
                'action': 'RESET',
                'expected_state': 'NOT_FINISHED',
                'expected_level_delta': 0,
                'expected_win': False,
                'expected_outcome': 'Playable frame appears.'
            },
            {
                'action': 'ACTION1',
                'expected_state': 'WIN',
                'expected_level_delta': 1,
                'expected_win': True,
                'expected_outcome': 'Fake runtime wins on ACTION1.'
            },
        ],
        'stop_on_mismatch': True,
        'reflection': 'Use sequence only when expectations are explicit.'
    }
"""
    next_observations, rewards, dones, infos = manager.step([f"<python>{program}</python>"])
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
    assert infos[0]["won"] is True
    assert infos[0]["actions_executed"] == 2
    assert [a["action"] for a in infos[0]["action_sequence_executed"]] == ["RESET", "ACTION1"]
    assert len(infos[0]["monitor_reports"]) == 2
    assert infos[0]["monitor_reports"][0]["expectation_met"] is True
    assert infos[0]["monitor_reports"][1]["expectation_met"] is True
    assert infos[0]["sequence_stopped_reason"] == "terminal"
    assert "monitor_reports" in next_observations["anchor"][0]["monitor_report"]


def test_arc_agi_3_official_multi_action_stops_on_mismatch(monkeypatch):
    _install_fake_arc_runtime(monkeypatch)
    cfg = _config(mode="official", max_action_sequence_len=3)
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=cfg.env)
    manager = ArcAGI3EnvironmentManager(env, arc_agi_3_projection, cfg)
    manager.reset([_official_task("ls20")])

    program = """def choose_action(frames, latest_frame):
    return {
        'hypothesis': 'The first expectation is intentionally wrong.',
        'plan': 'Monitor should stop before the second action.',
        'action_sequence': [
            {'action': 'RESET', 'expected_state': 'WIN', 'expected_level_delta': 1, 'expected_win': True},
            {'action': 'ACTION1', 'expected_state': 'WIN', 'expected_level_delta': 1, 'expected_win': True},
        ],
        'stop_on_mismatch': True,
    }
"""
    _next_observations, rewards, dones, infos = manager.step([f"<python>{program}</python>"])
    assert infos[0]["actions_executed"] == 1
    assert infos[0]["sequence_stopped_reason"] == "mismatch"
    assert infos[0]["monitor_reports"][0]["expectation_met"] is False
    assert infos[0]["won"] is False



def test_arc_agi_3_official_visual_dsl_and_context_compression(monkeypatch):
    _install_fake_arc_runtime_with_visual_change(monkeypatch)
    cfg = _config(mode="official", max_action_sequence_len=3)
    cfg.env.history_length = 2
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=cfg.env)
    manager = ArcAGI3EnvironmentManager(env, arc_agi_3_projection, cfg)
    observations, _infos = manager.reset([_official_task("ls20")])
    assert "expected_frame_changed" in observations["text"][0]
    assert "expected_visual_predicates" in observations["text"][0]
    assert "Compact summary:" in observations["text"][0]

    reset_program = """def choose_action(frames, latest_frame):
    return {'action':'RESET','expected_state':'NOT_FINISHED','expected_level_delta':0,'expected_win':False}
"""
    manager.step([f"<python>{reset_program}</python>"])

    visual_program = """def choose_action(context):
    assert 'latest_frame_summary' in context
    assert 'frame_summaries' in context
    return {
        'action':'ACTION1',
        'hypothesis':'color 2 object moves right',
        'plan':'press ACTION1 and verify frame diff',
        'expected_state':'NOT_FINISHED',
        'expected_frame_changed': True,
        'expected_changed_cell_count': {'eq': 2},
        'expected_cell': {'x':1,'y':1,'color':2,'layer':0,'when':'after'},
        'expected_color_delta': {'color':2,'delta':0,'layer':0},
        'expected_object_moved': {'color':2,'dx':1,'dy':0,'layer':0},
        'expected_visual_predicates': [
            {'op':'cell_changed','x':0,'y':1,'from':2,'to':0,'layer':0},
            {'op':'cell_changed','x':1,'y':1,'from':0,'to':2,'layer':0},
            {'op':'changed_cell_count','eq':2}
        ],
    }
"""
    next_observations, rewards, dones, infos = manager.step([f"<python>{visual_program}</python>"])
    report = infos[0]["monitor_report"]
    assert report["expectation_met"] is True
    assert report["actual"]["frame_changed"] is True
    assert report["actual"]["changed_cell_count"] == 2
    assert report["visual_checks"]
    assert not report["mismatches"]
    assert "monitor_summary" in infos[0]
    assert "Monitor summary:" in next_observations["text"][0]


def test_arc_agi_3_official_visual_mismatch_stops_sequence(monkeypatch):
    _install_fake_arc_runtime_with_visual_change(monkeypatch)
    cfg = _config(mode="official", max_action_sequence_len=3)
    env = build_arc_agi_3_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=cfg.env)
    manager = ArcAGI3EnvironmentManager(env, arc_agi_3_projection, cfg)
    manager.reset([_official_task("ls20")])

    program = """def choose_action(frames, latest_frame):
    return {
        'action_sequence': [
            {'action':'RESET','expected_visual_predicates':[{'op':'cell_equals','x':0,'y':0,'color':99,'layer':0,'when':'after'}]},
            {'action':'ACTION1','expected_frame_changed':True},
        ],
        'stop_on_mismatch': True,
    }
"""
    _next_observations, rewards, dones, infos = manager.step([f"<python>{program}</python>"])
    assert infos[0]["actions_executed"] == 1
    assert infos[0]["sequence_stopped_reason"] == "mismatch"
    assert infos[0]["monitor_reports"][0]["expectation_met"] is False
    assert any("expected_visual_predicates" in m["field"] for m in infos[0]["monitor_reports"][0]["mismatches"])
