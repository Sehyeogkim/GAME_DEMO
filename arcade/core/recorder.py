"""Buffers rows for one round and writes the episode folder; owns the training/other split rule."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import h5py
import numpy as np

GAME1016 = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = GAME1016 / "runs" / "arcade"
MAX_FRAMES = 2400
MIN_TRAINING_FRAMES = 5
OTHER_REASONS = ("game_fun", "failed_attempt", "assisted", "too_short", "setup", "idle")


def split_chunk(chunk):
    """The one place the training/other decision is made. Returns (split, other_reason)."""
    if not chunk["trainable"]:
        return "other", chunk["declared_reason"] or "game_fun"
    if chunk["script_frames"] or chunk["assist_frames"]:
        return "other", "assisted"
    if not chunk["success"]:
        return "other", "failed_attempt"
    if chunk["end"] - chunk["start"] < MIN_TRAINING_FRAMES:
        return "other", "too_short"
    return "training", None


def build_chunks(labels, results, sources, assists, active):
    chunks = []
    start = 0
    count = len(labels)
    for row in range(count):
        last = row == count - 1
        closes = results[row] is not None or last or labels[row + 1] != labels[row]
        if not closes:
            continue
        label, result = labels[row], results[row]
        end = row + 1
        chunk = {
            "index": len(chunks), "start": start, "end": end,
            "skill": label.skill, "instruction": label.instruction, "trainable": label.trainable,
            "declared_reason": None if label.trainable else label.other_reason,
            "success": bool(result.success) if result is not None else False,
            "closed_by": "result" if result is not None else ("episode_end" if last else "label_change"),
            "evidence": result.evidence if result is not None else {},
            "operator_frames": int(np.sum(sources[start:end] == 0)),
            "script_frames": int(np.sum(sources[start:end] != 0)),
            "assist_frames": int(np.sum(assists[start:end])),
            "active_ratio": float(np.mean(active[start:end])),
        }
        chunk["split"], chunk["other_reason"] = split_chunk(chunk)
        del chunk["declared_reason"]
        chunks.append(chunk)
        start = end
    return chunks


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class EpisodeRecorder:
    def __init__(self, runs_dir, game_name, metadata, model_xml, num_objects, max_frames=MAX_FRAMES):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.episode_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.root = Path(runs_dir or DEFAULT_RUNS) / game_name
        self.metadata = dict(metadata, episode_id=self.episode_id,
                             created_utc=datetime.now(timezone.utc).isoformat())
        self.model_xml = model_xml
        self.num_objects = num_objects
        self.max_frames = max_frames
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def append(self, observation, action, label, result, phase_index, score):
        self.rows.append((observation, action, label, result, phase_index, score))

    def chunks(self):
        if not self.rows:
            return []
        labels = [row[2] for row in self.rows]
        results = [row[3] for row in self.rows]
        sources = np.array([row[1]["source"] for row in self.rows])
        assists = np.array([row[1]["assist"] for row in self.rows])
        active = np.array([row[1]["operator_active"] for row in self.rows])
        return build_chunks(labels, results, sources, assists, active)

    def live_summary(self):
        chunks = self.chunks()
        return {"frames": len(self.rows),
                "training_chunks": sum(c["split"] == "training" for c in chunks[:-1]),
                "other_chunks": sum(c["split"] == "other" for c in chunks[:-1])}

    def close(self, end_reason, final_score=0.0):
        if not self.rows:
            return None
        chunks = self.chunks()
        rows = self.rows
        count = len(rows)
        partial = self.root / f".{self.episode_id}.partial"
        final = self.root / self.episode_id
        partial.mkdir(parents=True, exist_ok=False)
        chunk_index = np.zeros(count, np.int32)
        for chunk in chunks:
            chunk_index[chunk["start"]:chunk["end"]] = chunk["index"]
        with h5py.File(partial / "episode.hdf5", "w") as h5:
            images = h5.create_group("obs/images")
            jpeg = h5py.vlen_dtype(np.dtype("uint8"))
            for camera in rows[0][0]["images"]:
                dataset = images.create_dataset(camera, (count,), dtype=jpeg)
                for i, row in enumerate(rows):
                    dataset[i] = np.frombuffer(row[0]["images"][camera], np.uint8)
            for key in ("joint_pos", "joint_vel", "gripper_width", "eef_pos", "eef_quat"):
                h5.create_dataset(f"obs/{key}", data=np.array([r[0]["state"][key] for r in rows], np.float32))
            for key in ("command", "eef_target_pos", "eef_target_quat", "gripper", "joint_target"):
                h5.create_dataset(f"action/{key}", data=np.array([r[1][key] for r in rows], np.float32))
            h5.create_dataset("action/source", data=np.array([r[1]["source"] for r in rows], np.uint8))
            h5.create_dataset("action/assist", data=np.array([r[1]["assist"] for r in rows], bool))
            h5.create_dataset("action/operator_active", data=np.array([r[1]["operator_active"] for r in rows], bool))
            for key in ("qpos", "qvel", "mj_state"):
                h5.create_dataset(f"privileged/{key}", data=np.array([r[0][key] for r in rows], np.float64),
                                  compression="gzip", compression_opts=4)
            h5.create_dataset("privileged/objects",
                              data=np.array([r[0]["objects"] for r in rows], np.float64).reshape(count, self.num_objects, 7))
            h5.create_dataset("time/timestamp", data=np.array([r[0]["timestamp"] for r in rows], np.float64))
            h5.create_dataset("time/wall_time", data=np.array([r[0]["wall_time"] for r in rows], np.float64))
            h5.create_dataset("time/frame_index", data=np.arange(count, dtype=np.int64))
            h5.create_dataset("game/phase", data=np.array([r[4] for r in rows], np.uint8))
            h5.create_dataset("game/score", data=np.array([r[5] for r in rows], np.float32))
            h5.create_dataset("chunk/index", data=chunk_index)
            h5.attrs["schema_version"] = self.metadata["schema_version"]
            h5.attrs["episode_id"] = self.episode_id
        summary = {
            "chunks": len(chunks),
            "training_chunks": sum(c["split"] == "training" for c in chunks),
            "other_chunks": sum(c["split"] == "other" for c in chunks),
            "training_frames": sum(c["end"] - c["start"] for c in chunks if c["split"] == "training"),
            "other_frames": sum(c["end"] - c["start"] for c in chunks if c["split"] == "other"),
            "by_skill": {},
        }
        for chunk in chunks:
            entry = summary["by_skill"].setdefault(chunk["skill"], {"training": 0, "other": 0})
            entry[chunk["split"]] += 1
        metadata = dict(self.metadata, end_reason=end_reason, num_frames=count,
                        duration_s=count / self.metadata["control_hz"], final_score=final_score, summary=summary)
        (partial / "chunks.json").write_text(json.dumps(_json_safe({"chunks": chunks}), indent=1))
        (partial / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=1))
        (partial / "model.xml").write_text(self.model_xml)
        files = {}
        for path in sorted(partial.iterdir()):
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        (partial / "manifest.json").write_text(json.dumps({"episode_id": self.episode_id, "files": files}, indent=1))
        partial.rename(final)
        self.rows = []
        return {"episode_id": self.episode_id, "path": str(final), "frames": count,
                "end_reason": end_reason, **summary}
