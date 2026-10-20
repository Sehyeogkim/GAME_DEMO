"""One game round: physics, rendering, chunk labelling and recording, shared by server and tools."""
import hashlib
import io
import os
import sys
import time

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux; Windows and desktop Linux use the default (glfw)
import mujoco
import numpy as np
from PIL import Image

from . import scene
from .recorder import EpisodeRecorder
from .robot import CONTROL_HZ, DT, PandaController, ACTION_SCALE, STATE_ORDER, COMMAND_ORDER

SCHEMA_VERSION = "wemine.arcade.v1"


def jpeg_bytes(image, quality=90):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


class GameSession:
    def __init__(self, game, render_player=True, render_policy=True, record=True, runs_dir=None,
                 operator_kind="human", input_device="keyboard_mouse"):
        self.game = game
        self.xml = scene.build_model_xml(game.objects_xml(), game.decor_xml(), game.assets_xml(),
                                         game.player_camera, model_name=f"arcade_{game.name}")
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.robot = PandaController(self.model, self.data, game.bounds_low, game.bounds_high,
                                     game.lock_roll_pitch)
        game.model, game.data, game.robot = self.model, self.data, self.robot
        self.render_player = render_player
        self.render_policy = render_policy
        self.policy_renderer = mujoco.Renderer(self.model, *POLICY_HW) if render_policy else None
        self.player_renderer = mujoco.Renderer(self.model, *PLAYER_HW) if render_player else None
        self.policy_option = mujoco.MjvOption()
        self.player_option = mujoco.MjvOption()
        for group in range(6):
            self.policy_option.geomgroup[group] = int(group in scene.POLICY_GROUPS)
            self.player_option.geomgroup[group] = int(group in scene.PLAYER_GROUPS)
        self.record = record
        self.runs_dir = runs_dir
        self.operator_kind = operator_kind
        self.input_device = input_device
        self.object_ids = [game.body_id(name) for name in game.object_bodies()]
        if any(i < 0 for i in self.object_ids):
            raise ValueError(f"unknown object body in {game.object_bodies()}")
        self.substeps = round(DT / self.model.opt.timestep)
        self.recorder = None
        self.last_player_frame = None
        self.seed = 0
        self.round_index = 0
        self.model_sha256 = hashlib.sha256(self.xml.encode()).hexdigest()

    # ----- round lifecycle -----
    def reset(self, seed=None):
        if self.recorder is not None:
            self.finish("reset")
        self.seed = int(seed if seed is not None else np.random.SeedSequence().entropy % (2 ** 31))
        game = self.game
        game.rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        self.robot.home(game.home_position, game.home_yaw, settle_steps=0)
        game.phase = game.phases[0]
        game.score = 0.0
        game.round_over = False
        game.round_end_reason = None
        game.pending_result = None
        game.message = ""
        game.frame = 0
        game.reset()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(300):  # let objects settle with the arm holding its home pose
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self.round_index += 1
        if self.record:
            self.recorder = EpisodeRecorder(self.runs_dir, self.game.name, self.metadata(), self.xml,
                                            len(self.object_ids))
        return self.seed

    def finish(self, reason):
        """Close the round's recording. Returns the save summary or None."""
        recorder, self.recorder = self.recorder, None
        if recorder is None:
            return None
        return recorder.close(reason, final_score=float(self.game.score))

    # ----- one control step -----
    def step(self, command, buttons=None):
        game = self.game
        buttons = buttons or {}
        label = game.label
        observation = self.observe() if self.recorder is not None else None
        self.last_observation = observation
        control = game.before_action(np.clip(np.asarray(command, float), -1, 1), buttons)
        executed = self.robot.apply_command(control.command)
        assist = bool(control.assist)
        for _ in range(self.substeps):
            if game.substep():
                assist = True
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[:] = 0
        game.frame += 1
        game.after_action()
        result, game.pending_result = game.pending_result, None
        if self.recorder is not None:
            active = bool(np.any(np.abs(np.asarray(command)[:6]) > 0.05)) or (np.asarray(command)[6] > 0) != (observation["prev_grip"] > 0.5)
            self.recorder.append(observation, {
                "command": executed,
                "eef_target_pos": self.robot.target.copy(),
                "eef_target_quat": self.robot.target_quat(),
                "gripper": np.array([self.robot.grip]),
                "joint_target": self.robot.joint_target.copy(),
                "source": 0 if control.source == "operator" else 1,
                "assist": assist,
                "operator_active": bool(active) and control.source == "operator",
            }, label, result, game.phases.index(game.phase) if game.phase in game.phases else 0, game.score)
            if len(self.recorder) >= min(game.max_round_frames, self.recorder.max_frames):
                game.end_round("max_frames")
        return result

    def observe(self):
        state = self.robot.policy_state()
        images = {}
        if self.policy_renderer is not None:
            for camera in ("scene", "wrist"):
                self.policy_renderer.update_scene(self.data, camera=camera, scene_option=self.policy_option)
                images[camera] = jpeg_bytes(self.policy_renderer.render())
        if self.player_renderer is not None:
            images["player"] = jpeg_bytes(self.render_player_frame(), quality=80)
        native = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, self.data, native, mujoco.mjtState.mjSTATE_INTEGRATION)
        objects = np.array([[*self.data.xpos[i], *self.data.xquat[i]] for i in self.object_ids]).reshape(-1, 7)
        return {"state": state, "images": images, "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                "mj_state": native, "objects": objects, "timestamp": float(self.data.time),
                "wall_time": time.time(), "prev_grip": self.robot.grip}

    def render_player_frame(self):
        self.player_renderer.update_scene(self.data, camera="player", scene_option=self.player_option)
        self.last_player_frame = self.player_renderer.render().copy()
        return self.last_player_frame

    def render_policy_views(self):
        out = {}
        for camera in ("scene", "wrist"):
            self.policy_renderer.update_scene(self.data, camera=camera, scene_option=self.policy_option)
            out[camera] = self.policy_renderer.render().copy()
        return out

    # ----- description -----
    def metadata(self):
        cameras = {}
        rendered = (["scene", "wrist"] if self.render_policy else []) + (["player"] if self.render_player else [])
        for name in rendered:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            height, width = PLAYER_HW if name == "player" else POLICY_HW
            fovy = float(self.model.cam_fovy[cid])
            cameras[name] = {
                "policy_input": name != "player",
                "resolution_wh": [width, height], "fovy_deg": fovy,
                "intrinsics_K": scene.camera_intrinsics(fovy, width, height),
                "mount": "world" if self.model.cam_bodyid[cid] == 0 else "hand",
                "pos": self.model.cam_pos[cid].tolist(), "quat_wxyz": self.model.cam_quat[cid].tolist(),
                "geom_groups": list(scene.PLAYER_GROUPS if name == "player" else scene.POLICY_GROUPS),
                "encoding": "JPEG q80 4:4:4" if name == "player" else "JPEG q90 4:4:4",
            }
        game = self.game
        return {
            "schema_version": SCHEMA_VERSION, "game": game.name, "game_title": game.title,
            "game_version": game.version, "seed": self.seed, "round_index": self.round_index,
            "operator_kind": self.operator_kind, "input_device": self.input_device,
            "control_hz": CONTROL_HZ, "physics_dt": float(self.model.opt.timestep), "substeps": self.substeps,
            "mujoco_version": mujoco.__version__, "model_sha256": self.model_sha256,
            "robot": "Franka Emika Panda (MuJoCo Menagerie), fixed base, 7 arm joints + parallel gripper",
            "action_spec": {
                "command": {"order": COMMAND_ORDER, "range": [-1, 1], "frame": "world",
                            "scale_per_step": ACTION_SCALE.tolist() + [None],
                            "grip": "+1 close, -1 open (binary threshold at 0)",
                            "roll_pitch_locked": bool(game.lock_roll_pitch)},
                "eef_target": "absolute grasp_site target after applying command (pos m, quat wxyz)",
                "joint_target": "damped least-squares IK solution sent to Panda position actuators",
                "workspace_bounds": [list(game.bounds_low), list(game.bounds_high)],
            },
            "state_spec": {"order": STATE_ORDER, "dim": 22},
            "cameras": cameras,
            "objects": game.object_bodies(),
            "phases": game.phases,
            "skills": game.skills,
            "timing": "row t: observation before action t, then action t, then labels/results after action t",
        }

    def close(self):
        if self.recorder is not None:
            self.finish("stop")
        for renderer in (self.policy_renderer, self.player_renderer):
            if renderer is not None:
                renderer.close()
        self.policy_renderer = self.player_renderer = None


POLICY_HW = (scene.POLICY_IMAGE[1], scene.POLICY_IMAGE[0])
PLAYER_HW = (scene.PLAYER_IMAGE[1], scene.PLAYER_IMAGE[0])
