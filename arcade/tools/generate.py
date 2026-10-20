"""Record autopilot episodes headlessly and verify each one.

Usage: python -m arcade.tools.generate --game claw_prize --episodes 20 [--seed-start 0] [--no-player]
"""
import argparse
import json
import time

from arcade.core.session import GameSession
from arcade.core.verify import verify_episode
from arcade.games import load_game


def run(game_name, episodes, seed_start=0, render_player=True, runs_dir=None, quiet=False):
    game = load_game(game_name)
    session = GameSession(game, render_player=render_player, operator_kind="autopilot",
                          input_device="autopilot", runs_dir=runs_dir)
    results = []
    for offset in range(episodes):
        seed = seed_start + offset
        started = time.time()
        session.reset(seed=seed)
        while not game.round_over:
            command, buttons = game.autopilot()
            session.step(command, buttons)
        saved = session.finish(game.round_end_reason or "stop")
        report = verify_episode(saved["path"])
        row = {"seed": seed, "episode": saved["episode_id"], "end_reason": saved["end_reason"],
               "frames": saved["frames"], "training_chunks": saved["training_chunks"],
               "other_chunks": saved["other_chunks"], "by_skill": saved["by_skill"],
               "verified": report["ok"], "errors": report["errors"][:3],
               "seconds": round(time.time() - started, 1)}
        results.append(row)
        if not quiet:
            print(json.dumps(row), flush=True)
    session.close()
    complete = sum(r["end_reason"] == "round_complete" for r in results)
    summary = {"game": game_name, "episodes": len(results), "round_complete": complete,
               "verified": sum(r["verified"] for r in results),
               "training_chunks": sum(r["training_chunks"] for r in results),
               "other_chunks": sum(r["other_chunks"] for r in results)}
    print(json.dumps({"summary": summary}))
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--no-player", action="store_true", help="skip the player-view render to save time")
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()
    run(args.game, args.episodes, args.seed_start, not args.no_player, args.runs_dir)


if __name__ == "__main__":
    main()
