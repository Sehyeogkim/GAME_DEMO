"""Local single-player server for one arcade game. Every round is recorded and saved automatically.

Usage: python -m arcade.core.server --game claw_prize [--port 8802]
"""
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import io
import json
import logging
from pathlib import Path
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from arcade.games import GAMES, load_game
from .robot import DT
from .session import GameSession

STATIC = Path(__file__).with_name("static")
log = logging.getLogger("arcade")


class Arcade:
    """Owns the session; all MuJoCo work happens on the event-loop thread."""

    def __init__(self, game_name):
        self.game = load_game(game_name)
        self.session = GameSession(self.game, render_player=True, operator_kind="human")
        self.socket = None
        self.running = False
        self.paused = False
        self.command = np.zeros(7)
        self.buttons = {}
        self.last_saved = None
        self.saved_count = 0
        self.session.reset()

    def info(self):
        g = self.game
        return {"name": g.name, "title": g.title, "tagline": g.tagline, "controls": g.controls,
                "action_button": g.action_button, "skills": g.skills, "phases": g.phases,
                "max_round_seconds": g.max_round_frames * DT}

    def status(self):
        g, s = self.game, self.session
        recorder = s.recorder
        return {"phase": g.phase, "score": g.score, "message": g.message, "hud": g.hud(),
                "label": {"skill": g.label.skill, "instruction": g.label.instruction,
                          "trainable": g.label.trainable,
                          "other_reason": None if g.label.trainable else g.label.other_reason},
                "round": s.round_index, "seed": s.seed, "running": self.running, "paused": self.paused,
                "round_over": g.round_over, "recording": recorder.live_summary() if recorder else None,
                "elapsed_s": round(len(recorder) * DT, 2) if recorder else 0.0,
                "last_saved": self.last_saved, "saved_count": self.saved_count,
                "grip_closed": s.robot.grip > 0.5}

    def release_input(self):
        self.command = np.zeros(7)
        self.command[6] = 1.0 if self.session.robot.grip > 0.5 else -1.0
        self.buttons = {}

    def save(self, reason):
        saved = self.session.finish(reason)
        if saved:
            self.last_saved = saved
            self.saved_count += 1
            log.info("saved %s", json.dumps(saved))
        return saved

    def new_round(self):
        if self.session.recorder is not None and len(self.session.recorder):
            self.save("reset")
        self.session.reset()
        self.release_input()
        self.running = True
        self.paused = False

    def player_jpeg(self):
        observation = getattr(self.session, "last_observation", None)
        if observation is not None and "player" in observation["images"]:
            return observation["images"]["player"]
        frame = self.session.render_player_frame()
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()

    async def send(self, payload):
        if self.socket is not None:
            await self.socket.send_text(json.dumps(payload, default=float))

    async def loop(self):
        next_tick = time.perf_counter()
        while True:
            next_tick += DT
            try:
                if self.socket is not None:
                    if self.running and not self.paused and not self.game.round_over:
                        self.session.step(self.command, self.buttons)
                        self.buttons = {}
                        if self.game.round_over:
                            saved = self.save(self.game.round_end_reason or "round_complete")
                            self.running = False
                            await self.send({"type": "round_over", "saved": saved, "message": self.game.message,
                                             "score": self.game.score})
                    else:
                        self.session.last_observation = None
                    image = base64.b64encode(self.player_jpeg()).decode()
                    await self.send({"type": "frame", "image": image, "status": self.status()})
            except Exception:  # keep serving; surface the error to the page
                log.exception("step failed")
                await self.send({"type": "error", "message": "Simulation error — see server log."})
                self.running = False
            await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))
            if time.perf_counter() - next_tick > 0.5:
                next_tick = time.perf_counter()


def create_app(game_name):
    arcade = Arcade(game_name)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(arcade.loop())
        yield
        task.cancel()
        arcade.save("stop")
        arcade.session.close()

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    async def info():
        return JSONResponse(arcade.info())

    @app.get("/api/status")
    async def status():
        return JSONResponse(json.loads(json.dumps(arcade.status(), default=float)))

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        await socket.accept()
        if arcade.socket is not None:
            await socket.send_text(json.dumps({"type": "error", "message": "Another player is connected."}))
            await socket.close()
            return
        arcade.socket = socket
        await socket.send_text(json.dumps({"type": "info", "info": arcade.info()}))
        try:
            while True:
                message = json.loads(await socket.receive_text())
                kind = message.get("type")
                if kind == "hello":
                    arcade.session.input_device = str(message.get("input_device", "keyboard_mouse"))[:40]
                elif kind == "input":
                    command = np.clip(np.asarray(message.get("command", [0] * 7), float), -1, 1)
                    if command.shape == (7,) and np.all(np.isfinite(command)):
                        arcade.command = command
                    for key, value in (message.get("buttons") or {}).items():
                        arcade.buttons[key] = arcade.buttons.get(key, False) or bool(value)
                elif kind == "start":
                    arcade.new_round()
                elif kind == "pause":
                    arcade.paused = True
                    arcade.release_input()
                elif kind == "resume":
                    arcade.paused = False
                elif kind == "end_round":
                    saved = arcade.save("stop")
                    arcade.running = False
                    await arcade.send({"type": "round_over", "saved": saved, "message": "Round ended.",
                                       "score": arcade.game.score})
        except WebSocketDisconnect:
            pass
        finally:
            arcade.socket = None
            arcade.release_input()
            if arcade.running:
                arcade.save("disconnect")
                arcade.running = False
                arcade.session.reset()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=sorted(GAMES))
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import uvicorn
    uvicorn.run(create_app(args.game), host=args.host, port=args.port or GAMES[args.game][2], log_level="warning")


if __name__ == "__main__":
    main()
