"""Check one episode folder against the wemine.arcade.v1 contract.

Usage: python -m arcade.core.verify runs/arcade/<game>/<episode_id> [...]
"""
import argparse
import io
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image

from .recorder import OTHER_REASONS, sha256_file, split_chunk

REQUIRED_FILES = ["metadata.json", "episode.hdf5", "chunks.json", "model.xml", "manifest.json"]
PER_FRAME = {
    "obs/joint_pos": (7,), "obs/joint_vel": (7,), "obs/gripper_width": (1,), "obs/eef_pos": (3,),
    "obs/eef_quat": (4,), "action/command": (7,), "action/eef_target_pos": (3,),
    "action/eef_target_quat": (4,), "action/gripper": (1,), "action/joint_target": (7,),
    "action/source": (), "action/assist": (), "action/operator_active": (), "time/timestamp": (),
    "time/wall_time": (), "time/frame_index": (), "game/phase": (), "game/score": (), "chunk/index": (),
}


def verify_episode(path):
    path = Path(path)
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)
        return condition

    for name in REQUIRED_FILES:
        if not check((path / name).exists(), f"missing {name}"):
            return {"episode": str(path), "ok": False, "errors": errors}
    manifest = json.loads((path / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        file = path / name
        check(file.exists() and file.stat().st_size == info["bytes"], f"{name}: size mismatch")
        check(file.exists() and sha256_file(file) == info["sha256"], f"{name}: sha256 mismatch")
    metadata = json.loads((path / "metadata.json").read_text())
    chunks = json.loads((path / "chunks.json").read_text())["chunks"]
    check(metadata.get("schema_version") == "wemine.arcade.v1", "schema_version")
    count = metadata["num_frames"]
    dt = 1.0 / metadata["control_hz"]
    with h5py.File(path / "episode.hdf5", "r") as h5:
        for key, shape in PER_FRAME.items():
            if check(key in h5, f"missing dataset {key}"):
                check(h5[key].shape == (count, *shape), f"{key}: shape {h5[key].shape} != {(count, *shape)}")
                if h5[key].dtype.kind == "f":
                    check(np.isfinite(h5[key][:]).all(), f"{key}: non-finite values")
        for camera, spec in metadata["cameras"].items():
            if not check(f"obs/images/{camera}" in h5, f"missing images {camera}"):
                continue
            images = h5[f"obs/images/{camera}"]
            check(images.shape == (count,), f"images {camera}: count")
            for index in sorted({0, count // 2, count - 1}):
                image = Image.open(io.BytesIO(images[index].tobytes()))
                check(list(image.size) == spec["resolution_wh"], f"images {camera}[{index}]: size {image.size}")
        objects = h5["privileged/objects"]
        check(objects.shape == (count, len(metadata["objects"]), 7), "privileged/objects shape")
        for key in ("qpos", "qvel", "mj_state"):
            check(h5[f"privileged/{key}"].shape[0] == count, f"privileged/{key} rows")
        timestamps = h5["time/timestamp"][:]
        check(np.allclose(np.diff(timestamps), dt, atol=1e-6), "timestamps not at control_hz")
        check(np.array_equal(h5["time/frame_index"][:], np.arange(count)), "frame_index not contiguous")
        command = h5["action/command"][:]
        check(np.all(np.abs(command) <= 1 + 1e-6), "command outside [-1, 1]")
        gripper = h5["action/gripper"][:]
        check(np.all(np.isin(gripper, [0.0, 1.0])), "gripper target not binary")
        check(np.all(h5["game/phase"][:] < len(metadata["phases"])), "phase index out of range")
        source = h5["action/source"][:]
        assist = h5["action/assist"][:]
        chunk_index = h5["chunk/index"][:]

    # Chunks tile [0, count) exactly and agree with the per-frame index.
    cursor = 0
    for i, chunk in enumerate(chunks):
        check(chunk["index"] == i, f"chunk {i}: index")
        check(chunk["start"] == cursor and chunk["end"] > chunk["start"], f"chunk {i}: not contiguous")
        cursor = chunk["end"]
        check(np.all(chunk_index[chunk["start"]:chunk["end"]] == i), f"chunk {i}: chunk/index disagrees")
        rule = split_chunk(dict(chunk, declared_reason=chunk["other_reason"] if not chunk["trainable"] else None))
        check((chunk["split"], chunk["other_reason"]) == rule, f"chunk {i}: split {chunk['split']} violates rule {rule}")
        check(chunk["split"] in ("training", "other"), f"chunk {i}: split value")
        check(chunk["other_reason"] is None or chunk["other_reason"] in OTHER_REASONS, f"chunk {i}: other_reason")
        if chunk["split"] == "training":
            window = slice(chunk["start"], chunk["end"])
            check(np.all(source[window] == 0) and not np.any(assist[window]), f"chunk {i}: training chunk has script/assist rows")
            check(chunk["success"] and chunk["skill"] in metadata["skills"], f"chunk {i}: training chunk not a successful declared skill")
    check(cursor == count, f"chunks cover {cursor} of {count} frames")
    summary = {
        "frames": count,
        "training_chunks": sum(c["split"] == "training" for c in chunks),
        "training_frames": sum(c["end"] - c["start"] for c in chunks if c["split"] == "training"),
        "other_chunks": sum(c["split"] == "other" for c in chunks),
        "other_reasons": sorted({c["other_reason"] for c in chunks if c["other_reason"]}),
    }
    return {"episode": str(path), "ok": not errors, "errors": errors, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+")
    args = parser.parse_args()
    failed = 0
    for episode in args.episodes:
        report = verify_episode(episode)
        failed += not report["ok"]
        print(json.dumps(report))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
