"""Closed-loop evaluation on new seeds: the policy drives every trainable chunk, autopilot drives the rest.

Usage:
  python -m arcade.tools.eval_policy --game claw_prize --policy runs/arcade/_models/.../policy.pt
  python -m arcade.tools.eval_policy --game claw_prize --baseline zero|random
Metric per trainable chunk attempt: success (physical check passed), failure (game closed it as failed),
or timeout (policy did not finish within --timeout frames; ends the episode).
"""
import argparse
import json

import numpy as np
import torch

from arcade.core.session import GameSession
from arcade.games import load_game
from arcade.tools.policy import BCPolicy, default_device, images_to_tensor, load_policy, numpy_state
from PIL import Image


class PolicyDriver:
    def __init__(self, path, device):
        self.policy, checkpoint = load_policy(path, device)
        self.size = checkpoint["image_size"]
        self.device = device
        self.queue = []

    def reset(self):
        self.queue = []

    def act(self, session, skill):
        views = session.render_policy_views()
        images = np.stack([np.asarray(Image.fromarray(views[c]).resize((self.size, self.size), Image.BILINEAR))
                           for c in ("scene", "wrist")])[None]
        state = torch.as_tensor(numpy_state(session.robot.policy_state())[None], device=self.device)
        skill_index = torch.as_tensor([self.policy.skills.index(skill)], device=self.device)
        with torch.no_grad():
            out = self.policy(images_to_tensor(images, self.device), state, skill_index)
        chunk = BCPolicy.to_command(out)[0].cpu().numpy()
        # Temporal ensemble: average the overlapping predictions for this step (ACT-style).
        self.queue = [q[1:] for q in self.queue if len(q) > 1] + [chunk]
        weights = np.exp(-0.25 * np.arange(len(self.queue)))[::-1]
        command = sum(w * q[0] for w, q in zip(weights, self.queue)) / weights.sum()
        command[6] = 1.0 if command[6] > 0 else -1.0
        return command


def run(args):
    game = load_game(args.game)
    session = GameSession(game, render_player=False, render_policy=True, record=False)
    rng = np.random.default_rng(args.seed_start)
    driver = PolicyDriver(args.policy, default_device()) if args.policy else None
    skills = set(driver.policy.skills) if driver else set(game.skills)
    totals = {"success": 0, "failure": 0, "timeout": 0}
    per_episode = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        session.reset(seed=seed)
        counts = {"success": 0, "failure": 0, "timeout": 0}
        chunk_frames = 0
        frames = 0
        if driver:
            driver.reset()
        while not game.round_over and frames < game.max_round_frames:
            label = game.label
            policy_turn = label.trainable and label.skill in skills
            if policy_turn:
                if args.baseline == "zero":
                    command = np.zeros(7)
                    command[6] = -1.0
                elif args.baseline == "random":
                    command = rng.uniform(-1, 1, 7)
                else:
                    command = driver.act(session, label.skill)
                chunk_frames += 1
                buttons = {}
            else:
                command, buttons = game.autopilot()
            result = session.step(command, buttons)
            frames += 1
            if policy_turn and result is not None:
                counts["success" if result.success else "failure"] += 1
                chunk_frames = 0
                if driver:
                    driver.reset()
            if policy_turn and chunk_frames >= args.timeout:
                counts["timeout"] += 1
                break
        for key in totals:
            totals[key] += counts[key]
        per_episode.append({"seed": seed, **counts, "frames": frames, "round_complete": game.round_end_reason == "round_complete"})
        if not args.quiet:
            print(json.dumps(per_episode[-1]), flush=True)
    attempts = sum(totals.values())
    summary = {"game": args.game, "controller": args.policy or args.baseline, "episodes": args.episodes,
               "seed_start": args.seed_start, "attempts": attempts, **totals,
               "success_rate": round(totals["success"] / attempts, 3) if attempts else 0.0,
               "rounds_complete": sum(e["round_complete"] for e in per_episode)}
    print(json.dumps({"summary": summary}))
    session.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--baseline", choices=["zero", "random"], default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100000, help="held-out seeds, far from generation seeds")
    parser.add_argument("--timeout", type=int, default=400)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not args.policy and not args.baseline:
        parser.error("give --policy or --baseline")
    run(args)


if __name__ == "__main__":
    main()
