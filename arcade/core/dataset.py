"""Load action chunks from recorded episodes as arrays ready for policy training.

Only policy-safe fields are returned: the two policy cameras, the 22-D robot state,
the skill id, and future commands inside the same chunk. privileged/* is never read.
"""
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .recorder import DEFAULT_RUNS
from .verify import verify_episode

POLICY_CAMERAS = ("scene", "wrist")
STATE_KEYS = ("joint_pos", "joint_vel", "gripper_width", "eef_pos", "eef_quat")


def find_episodes(game, runs_dir=None, operator_kinds=None):
    folder = Path(runs_dir or DEFAULT_RUNS) / game
    if not folder.exists():
        return []
    episodes = []
    for path in sorted(folder.iterdir()):
        if not path.is_dir() or path.name.startswith(".") or not (path / "metadata.json").exists():
            continue
        meta = json.loads((path / "metadata.json").read_text())
        if operator_kinds and meta.get("operator_kind") not in operator_kinds:
            continue
        episodes.append(path)
    return episodes


def decode(jpeg, size):
    image = Image.open(io.BytesIO(jpeg.tobytes())).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image)


def load_chunks(episodes, split="training", image_size=96, horizon=8, skills=None, verify=True):
    """Return per-frame samples from chunks with the given split ("training", "other" or None for all)."""
    images, states, skill_names, actions, valid, frame_refs, chunk_refs = [], [], [], [], [], [], []
    report = {"episodes": 0, "skipped_unverified": [], "chunks": 0, "frames": 0}
    for episode in episodes:
        episode = Path(episode)
        if verify and not verify_episode(episode)["ok"]:
            report["skipped_unverified"].append(episode.name)
            continue
        chunks = json.loads((episode / "chunks.json").read_text())["chunks"]
        selected = [c for c in chunks if (split is None or c["split"] == split) and (skills is None or c["skill"] in skills)]
        if not selected:
            continue
        report["episodes"] += 1
        with h5py.File(episode / "episode.hdf5", "r") as h5:
            state = np.concatenate([h5[f"obs/{k}"][:] for k in STATE_KEYS], axis=1)
            command = h5["action/command"][:]
            cams = {c: h5[f"obs/images/{c}"] for c in POLICY_CAMERAS}
            for chunk in selected:
                report["chunks"] += 1
                start, end = chunk["start"], chunk["end"]
                for t in range(start, end):
                    future = np.arange(t, t + horizon)
                    mask = future < end
                    future = np.minimum(future, end - 1)
                    images.append(np.stack([decode(cams[c][t], image_size) for c in POLICY_CAMERAS]))
                    states.append(state[t])
                    skill_names.append(chunk["skill"])
                    actions.append(command[future])
                    valid.append(mask)
                    frame_refs.append((episode.name, t))
                    chunk_refs.append(f"{episode.name}:{chunk['index']}")
                    report["frames"] += 1
    if not states:
        raise ValueError(f"no {split} chunks found in {len(episodes)} episodes")
    return {
        "images": np.stack(images),               # (N, cams, H, W, 3) uint8
        "state": np.asarray(states, np.float32),  # (N, 22)
        "skill": np.asarray(skill_names),         # (N,) str
        "actions": np.asarray(actions, np.float32),  # (N, horizon, 7)
        "valid": np.asarray(valid, bool),         # (N, horizon)
        "frame_refs": frame_refs, "chunk_refs": np.asarray(chunk_refs),
    }, report
