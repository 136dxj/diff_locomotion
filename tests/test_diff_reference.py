"""In-memory schema sentinels only; these are not reference motion trajectories."""

from io import BytesIO

import numpy as np
import pytest
import torch

from src.diff_loco.reference import ReferenceStates


def bank(n=3):
    frame = np.arange(4, dtype=np.float64)
    qpos = np.zeros((4, 7 + n))
    qpos[:, 0], qpos[:, 3], qpos[:, 7:] = frame, 1, frame[:, None] / 10
    return dict(joint_names=np.array([f"joint_{i}" for i in range(n)]), qpos=qpos,
                qvel=np.broadcast_to(frame[:, None], (4, 6 + n)).copy(), phase=frame / 4,
                command=frame[:, None] * np.array([0.1, -0.2, 0.3]), period=np.array(0.6),
                action_history=np.broadcast_to(
                    frame[:, None, None] / 4 + np.array([-0.5, 0.25])[None, :, None],
                    (4, 2, n)).copy())


def archive(data):
    stream = BytesIO()
    np.savez(stream, **data)
    stream.seek(0)
    return stream


@pytest.mark.parametrize("n", [12, 29])
def test_sample_keeps_frames_and_history_aligned_without_aliases(n):
    data = bank(n)
    refs = ReferenceStates(archive(data), data["joint_names"].tolist(), "cpu", 0.6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        sample = refs.sample(32)
        index = sample["qpos"][:, 0].long()
        assert index.unique().numel() > 1
        assert set(sample) == {"qpos", "qvel", "phase", "command", "action_history"}
        for key, value in sample.items():
            expected = torch.tensor(data[key], dtype=torch.float32)[index]
            torch.testing.assert_close(value, expected)
            value[0].fill_(99)
            torch.testing.assert_close(value[1:], expected[1:])
        torch.manual_seed(17)
        again = refs.sample(32)
    for key, value in again.items():
        torch.testing.assert_close(value, torch.tensor(data[key], dtype=torch.float32)[index])


def test_optional_history_and_empty_sampling():
    data = bank()
    del data["action_history"]
    data["period"] = np.array(0.6, dtype=np.float32)
    data["qpos"][:, 3] += 5e-5  # Inside the specified absolute quaternion tolerance.
    refs = ReferenceStates(archive(data), data["joint_names"].tolist(), "cpu", 0.6)
    assert torch.equal(refs.sample(8)["action_history"], torch.zeros(8, 2, 3))
    assert {key: tuple(value.shape) for key, value in refs.sample(0).items()} == {
        "qpos": (0, 10), "qvel": (0, 9), "phase": (0,),
        "command": (0, 3), "action_history": (0, 2, 3),
    }
    for count in (-1, 1.5, True):
        with pytest.raises(ValueError, match="count"):
            refs.sample(count)


@pytest.mark.parametrize("names,period", [
    (["joint_1", "joint_0", "joint_2"], 0.6),
    (["joint_0", "joint_1"], 0.6),
    (["joint_0", "joint_1", "other"], 0.6),
    (["joint_0", "joint_1", "joint_2"], 0.5),
])
def test_rejects_model_joint_or_period_mismatch(names, period):
    with pytest.raises(ValueError, match="joint|period"):
        ReferenceStates(archive(bank()), names, "cpu", period)


@pytest.mark.parametrize("field,value", [
    ("qpos", np.empty((0, 10))),
    ("qpos", np.zeros((4, 10))),  # Zero quaternion.
    ("qpos", np.tile([0, 0, 0, 1.0002, 0, 0, 0, 0, 0, 0], (4, 1))),
    ("qvel", np.zeros((3, 9))),
    ("qvel", np.full((4, 9), np.nan)),
    ("command", np.zeros((4, 2))),
    ("command", np.full((4, 3), np.inf)),
    ("command", np.full((4, 3), 1e100)),  # Finite input, float32 overflow.
    ("phase", np.zeros((4, 1))),
    ("phase", np.full(4, -0.01)),
    ("phase", np.ones(4)),
    ("phase", np.full(4, 1 - 1e-10)),  # Would round to one in float32.
    ("period", np.array([0.6])),
    ("period", np.array(0.0)),
    ("period", np.array(np.nan)),
    ("action_history", np.zeros((4, 3, 2))),
    ("action_history", np.full((4, 2, 3), 1.01)),
    ("action_history", np.full((4, 2, 3), -1.01)),
    ("joint_names", np.array(["joint_0", "joint_1", "joint_2"], dtype=object)),
    ("command", None),
])
def test_rejects_invalid_states_and_pickle_arrays(field, value):
    data = bank()
    if value is None:
        del data[field]
    else:
        data[field] = value
    with pytest.raises(ValueError):
        ReferenceStates(archive(data), ["joint_0", "joint_1", "joint_2"], "cpu", 0.6)
