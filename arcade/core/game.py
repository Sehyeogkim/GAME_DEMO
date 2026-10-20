"""Base class every arcade game implements.

A game only decides three things: what is in the scene, what the rules are, and
which skill the current frame belongs to. Recording, the training/other split,
rendering and networking are owned by the core so all games share one contract.

Per control step the session calls, in order:
  1. game.label            -> Label for this row (read before the action)
  2. game.before_action(command, buttons) -> StepControl
  3. physics substeps, calling game.substep() before each mj_step
  4. game.after_action()   -> may call self.finish_chunk(...) / self.end_round(...)
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass(frozen=True)
class Label:
    skill: str
    instruction: str
    trainable: bool
    other_reason: str = "game_fun"   # used when trainable is False: game_fun | setup | idle


@dataclass
class StepControl:
    """What actually drives the arm this step."""
    command: np.ndarray
    source: str = "operator"         # operator | game_script
    assist: bool = False             # game applies non-physical forces/teleports this step


@dataclass
class ChunkResult:
    success: bool
    evidence: dict = field(default_factory=dict)


class ArcadeGame:
    # --- identity, overridden by each game ---
    name = "base"
    title = "Base"
    version = "1"
    port = 8800
    tagline = ""
    skills = {}                      # skill id -> description, trainable skills only
    phases = ["play"]
    controls = []                    # extra (key, text) rows for the help panel
    action_button = None             # label of the game action button (Enter / right click), or None
    # Workspace for the gripper centre (grasp_site), metres.
    bounds_low = (0.15, -0.42, 0.752)
    bounds_high = (0.85, 0.42, 1.25)
    home_position = (0.40, 0.0, 0.98)
    home_yaw = 0.0
    lock_roll_pitch = True
    player_camera = None             # dict(pos, target, fovy) or None for the default
    max_round_frames = 2400

    def __init__(self):
        self.model = None
        self.data = None
        self.robot = None
        self.rng = np.random.default_rng(0)
        self.phase = self.phases[0]
        self.score = 0.0
        self.label = Label("idle", "", False, "idle")
        self.pending_result = None
        self.round_over = False
        self.round_end_reason = None
        self.message = ""
        self.frame = 0

    # --- scene, overridden ---
    def objects_xml(self):
        return ""

    def decor_xml(self):
        return ""

    def assets_xml(self):
        return ""

    def object_bodies(self):
        """Body names whose world poses are logged as privileged/objects."""
        return []

    # --- rules, overridden ---
    def reset(self):
        """Place objects using self.rng; set phase/label. Arm is already homed."""

    def before_action(self, command, buttons):
        return StepControl(command=np.asarray(command, float))

    def substep(self):
        """Called before every physics substep; apply game-only forces here (mark assist)."""

    def after_action(self):
        pass

    def hud(self):
        return {}

    def autopilot(self):
        """Scripted expert: return (command7, buttons) from privileged state, like a human would."""
        return np.zeros(7), {}

    # --- helpers for games ---
    def finish_chunk(self, success, **evidence):
        """Close the current chunk at this row (inclusive). Next row starts a new chunk."""
        self.pending_result = ChunkResult(bool(success), evidence)

    def end_round(self, reason="round_complete", message=""):
        self.round_over = True
        self.round_end_reason = reason
        if message:
            self.message = message

    def body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def geom_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def body_pos(self, name):
        return self.data.xpos[self.body_id(name)].copy()

    def body_quat(self, name):
        return self.data.xquat[self.body_id(name)].copy()

    def set_free_body(self, name, pos, quat=(1, 0, 0, 0)):
        """Teleport a freejoint body (only during reset; mid-round teleports must set assist)."""
        body = self.body_id(name)
        joint = self.model.body_jntadr[body]
        qadr = self.model.jnt_qposadr[joint]
        dadr = self.model.jnt_dofadr[joint]
        self.data.qpos[qadr:qadr + 7] = [*pos, *quat]
        self.data.qvel[dadr:dadr + 6] = 0

    def body_upright(self, name):
        """Cosine between the body's +z axis and world +z."""
        return float(self.data.xmat[self.body_id(name)].reshape(3, 3)[2, 2])

    def touching(self, body_a, body_b):
        a, b = self.body_id(body_a), self.body_id(body_b)
        for contact in self.data.contact[:self.data.ncon]:
            pair = (self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2])
            if a in pair and b in pair:
                return True
        return False

    def finger_contacts(self, body):
        """(left, right) finger contact with `body`: the physical grasp evidence."""
        return self.touching(body, "left_finger"), self.touching(body, "right_finger")

    def grasped(self, body):
        left, right = self.finger_contacts(body)
        return bool(left and right and self.robot.grip > 0.5)

    # Autopilot helpers: a proportional move toward a point, shaped like human input.
    def steer_to(self, point, gain=1.0, yaw=None, tolerance=0.01):
        from .robot import ACTION_SCALE
        command = np.zeros(7)
        error = np.asarray(point, float) - self.robot.target
        command[:3] = np.clip(error / ACTION_SCALE[:3] * gain * 0.5, -1, 1)
        if yaw is not None:
            yaw_error = (yaw - self.robot.euler[2] + np.pi) % (2 * np.pi) - np.pi
            command[5] = np.clip(yaw_error / ACTION_SCALE[5] * 0.5, -1, 1)
        command[6] = 1.0 if self.robot.grip > 0.5 else -1.0
        # Reached when the commanded target is there and the real gripper has caught up.
        point = np.clip(np.asarray(point, float), self.robot.bounds_low, self.robot.bounds_high)
        reached = (np.linalg.norm(point - self.robot.target) < tolerance
                   and np.linalg.norm(point - self.robot.eef_pos()) < max(tolerance, 0.02))
        return command, reached
