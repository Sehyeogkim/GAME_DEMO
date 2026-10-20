"""Train a behaviour-cloning policy on one game's chunks (default: split == "training" only).

Usage: python -m arcade.tools.train_bc --game claw_prize [--steps 4000] [--split training]
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from arcade.core.dataset import find_episodes, load_chunks
from arcade.core.recorder import DEFAULT_RUNS
from arcade.tools.policy import BCPolicy, default_device, images_to_tensor, save_policy


def augment(images, pad=6):
    """Random shift: pad then crop back, per sample."""
    b, c, ch, h, w = images.shape
    padded = F.pad(images.reshape(b, c * ch, h, w), (pad, pad, pad, pad), mode="replicate")
    out = torch.empty((b, c * ch, h, w), device=images.device)
    offsets = torch.randint(0, 2 * pad + 1, (b, 2))
    for i in range(b):
        y, x = offsets[i]
        out[i] = padded[i, :, y:y + h, x:x + w]
    return out.view(b, c, ch, h, w)


def bc_loss(out, actions, valid):
    motion = F.mse_loss(torch.tanh(out[..., :6]), actions[..., :6], reduction="none").mean(-1)
    grip = F.binary_cross_entropy_with_logits(out[..., 6], (actions[..., 6] > 0).float(), reduction="none")
    per_step = (motion + 0.5 * grip) * valid
    return per_step.sum() / valid.sum().clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--split", default="training", help='"training" (default), "other", or "all"')
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--operator", default=None, help="human | autopilot (default: both)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    episodes = find_episodes(args.game, operator_kinds=[args.operator] if args.operator else None)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    split = None if args.split == "all" else args.split
    data, report = load_chunks(episodes, split=split, image_size=args.image_size, horizon=args.horizon)
    skills = sorted(set(data["skill"].tolist()))
    # Hold out whole chunks for validation, never single frames from a training chunk.
    chunk_ids = np.unique(data["chunk_refs"])
    rng.shuffle(chunk_ids)
    val_chunks = set(chunk_ids[:max(1, len(chunk_ids) // 10)])
    is_val = np.array([c in val_chunks for c in data["chunk_refs"]])
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]

    device = default_device()
    policy = BCPolicy(skills, args.horizon).to(device)
    policy.state_mean.copy_(torch.as_tensor(data["state"][train_idx].mean(0)))
    policy.state_std.copy_(torch.as_tensor(data["state"][train_idx].std(0) + 1e-3))
    skill_index = np.array([skills.index(s) for s in data["skill"]])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    def batch(indices, train):
        images = images_to_tensor(data["images"][indices], device)
        if train:
            images = augment(images)
        return (images, torch.as_tensor(data["state"][indices], device=device),
                torch.as_tensor(skill_index[indices], device=device),
                torch.as_tensor(data["actions"][indices], device=device),
                torch.as_tensor(data["valid"][indices], device=device).float())

    def validate():
        policy.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(val_idx), 256):
                images, state, skill, actions, valid = batch(val_idx[start:start + 256], False)
                losses.append(bc_loss(policy(images, state, skill), actions, valid).item() * len(images))
        policy.train()
        return sum(losses) / max(1, len(val_idx))

    log = []
    started = time.time()
    for step in range(1, args.steps + 1):
        images, state, skill, actions, valid = batch(rng.choice(train_idx, args.batch), True)
        loss = bc_loss(policy(images, state, skill), actions, valid)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 250 == 0 or step == 1:
            entry = {"step": step, "train_loss": round(loss.item(), 5), "val_loss": round(validate(), 5),
                     "seconds": round(time.time() - started, 1)}
            log.append(entry)
            print(json.dumps(entry), flush=True)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path(args.out or DEFAULT_RUNS / "_models" / args.game / f"{stamp}-{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    info = {"game": args.game, "split": args.split, "skills": skills, "image_size": args.image_size,
            "horizon": args.horizon, "steps": args.steps, "device": device, "data": report,
            "train_frames": int(len(train_idx)), "val_frames": int(len(val_idx)),
            "episodes": [p.name for p in episodes], "log": log}
    save_policy(out / "policy.pt", policy, {"image_size": args.image_size, "info": info})
    (out / "train_log.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({"saved": str(out / "policy.pt"), "frames": report["frames"], "chunks": report["chunks"]}))


if __name__ == "__main__":
    main()
