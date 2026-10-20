"""Start every ready game server plus a hub page at http://127.0.0.1:8800.

Usage: python -m arcade.tools.launch [--games claw_prize,domino_run]
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from arcade.games import GAMES, load_game

HUB_PORT = 8800
GAME1016 = Path(__file__).resolve().parents[2]
RUNS = GAME1016 / "runs" / "arcade"


def ready_games(requested=None):
    names = requested or list(GAMES)
    ready = []
    for name in names:
        try:
            load_game(name)
            ready.append(name)
        except Exception as error:  # a game still under construction should not block the others
            print(f"skip {name}: {type(error).__name__}: {error}", file=sys.stderr)
    return ready


def episode_counts():
    counts = {}
    for name in GAMES:
        folder = RUNS / name
        episodes = [p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")] if folder.exists() else []
        training = 0
        human = 0
        for episode in episodes:
            try:
                meta = json.loads((episode / "metadata.json").read_text())
                training += meta["summary"]["training_chunks"]
                human += meta["operator_kind"] == "human"
            except (OSError, KeyError, ValueError):
                pass
        counts[name] = {"episodes": len(episodes), "human_episodes": human, "training_chunks": training}
    return counts


def hub_app(games):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI()
    infos = {}
    for name in games:
        game = load_game(name)
        infos[name] = {"title": game.title, "tagline": game.tagline, "port": GAMES[name][2],
                       "skills": list(game.skills)}

    @app.get("/api/games")
    async def api_games():
        return JSONResponse({"games": infos, "counts": episode_counts()})

    @app.get("/")
    async def index():
        return HTMLResponse(HUB_HTML)

    return app


HUB_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>WEMINE Arcade</title>
<style>
:root{--bg:#0f1220;--panel:#171b2e;--line:#2a3050;--text:#e8eaf6;--muted:#9aa0c0;--accent:#ff5fa8;--train:#3ddc97;color-scheme:dark}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,sans-serif}
header{padding:28px 24px 8px;max-width:1100px;margin:auto}h1{margin:0;font-size:30px}header p{color:var(--muted);margin:4px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;padding:16px 24px 32px;max-width:1100px;margin:auto}
a.card{display:block;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;color:inherit;text-decoration:none}
a.card:hover{border-color:var(--accent)}a.card h2{margin:0 0 4px;font-size:19px}a.card p{margin:0 0 10px;color:var(--muted);font-size:14px}
.skill{display:inline-block;font-size:12px;color:var(--train);background:rgba(61,220,151,.12);border-radius:999px;padding:2px 8px;margin:0 4px 4px 0}
.stats{font-size:13px;color:var(--muted);margin-top:6px}
</style></head><body><header><h1>WEMINE Arcade</h1>
<p>Each game records every round. Green skills are the action chunks kept for robot training; everything else is saved as "other".</p></header>
<div class="grid" id="grid"></div>
<script>
fetch('/api/games').then(r=>r.json()).then(({games,counts})=>{
  const grid=document.getElementById('grid');
  for(const [name,g] of Object.entries(games)){
    const a=document.createElement('a');a.className='card';a.href=`http://127.0.0.1:${g.port}/`;
    const c=counts[name]||{episodes:0,human_episodes:0,training_chunks:0};
    a.innerHTML='<h2></h2><p></p><div class="skills"></div><div class="stats"></div>';
    a.querySelector('h2').textContent=g.title;a.querySelector('p').textContent=g.tagline;
    for(const s of g.skills){const b=document.createElement('span');b.className='skill';b.textContent=s;a.querySelector('.skills').append(b)}
    a.querySelector('.stats').textContent=`${c.episodes} episodes (${c.human_episodes} human) · ${c.training_chunks} training chunks`;
    grid.append(a);
  }
});
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=None, help="comma-separated subset")
    args = parser.parse_args()
    games = ready_games(args.games.split(",") if args.games else None)
    children = []
    for name in games:
        children.append(subprocess.Popen([sys.executable, "-m", "arcade.core.server", "--game", name], cwd=GAME1016))
        print(f"{name:14s} http://127.0.0.1:{GAMES[name][2]}/")
    print(f"{'hub':14s} http://127.0.0.1:{HUB_PORT}/")
    import uvicorn
    try:
        uvicorn.run(hub_app(games), host="127.0.0.1", port=HUB_PORT, log_level="warning")
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
