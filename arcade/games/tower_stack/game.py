"""Tower Stack: stack blocks on the base pad (trainable), then survive a wind gust (game fun).

Trainable skill `stack_block`: pick a free block and place it centred on top of the current
tower (the first block goes on the pad). After every successful placement the game blows a
short lateral gust at the tower (xfrc_applied, marked assist). A well-centred tower survives
and earns a bonus; a sloppy one topples and has to be rebuilt.
"""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label, StepControl

TABLE_Z = 0.74
BLOCKS = [
    # name, half size (cube), rgba
    ("block_red", 0.025, "0.90 0.26 0.22 1"),
    ("block_blue", 0.024, "0.22 0.48 0.92 1"),
    ("block_yellow", 0.024, "0.97 0.78 0.18 1"),
    ("block_green", 0.023, "0.30 0.76 0.38 1"),
    ("block_purple", 0.022, "0.62 0.36 0.85 1"),
]
HALF = {name: half for name, half, _ in BLOCKS}
PAD_HALF = 0.034
GOAL_HEIGHT = 4

STACK = Label("stack_block", "Pick up a block and place it centred on top of the tower.", True)
WIND = Label("wind_gust", "Hold still while the wind tests the tower.", False, "game_fun")

# Placement success thresholds.
MAX_OFFSET_M = 0.02
MIN_UPRIGHT = 0.97
HEIGHT_TOL_M = 0.006
STABLE_S = 0.5
CLEARANCE_M = 0.02          # gripper centre must be this far above the placed block's top face
SETTLE_SPEED = 0.03         # m/s, block counted as resting below this
PLACE_TIMEOUT_S = 4.0
# Tower integrity (a block still counts as part of the standing tower).
TOWER_Z_TOL = 0.012
TOWER_XY_TOL = 0.035
TOWER_UPRIGHT = 0.90
# Wind gust.
GUST_FRAMES = 20            # 1.0 s ramp up and down
WIND_MIN_FRAMES = 30        # 1.5 s fun chunk at least
WIND_MAX_FRAMES = 50
WIND_FORCE_N = (0.12, 0.15)  # peak lateral force on the top block, scaled down for lower blocks


def wrap_quarter(angle):
    """Wrap to [-pi/4, pi/4): cubes look the same every 90 degrees of yaw."""
    return (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4


def wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class TowerStack(ArcadeGame):
    name = "tower_stack"
    title = "Tower Stack"
    version = "1"
    port = 8804
    tagline = "Stack blocks into a tower, then hold on while the wind blows."
    skills = {"stack_block": (
        "Pick a block and release it on top of the tower (first block on the base pad): resting on the "
        "previous top block or pad, horizontal offset from the tower axis < 2 cm, face-up cosine > 0.97, "
        "gripper clear of the block, stable for 0.5 s.")}
    phases = ["stack", "wind", "done"]
    controls = [("Goal", "Stack 4 blocks on the glowing pad"),
                ("Wind", "After each block a gust hits the tower: centred stacks survive")]
    home_position = (0.42, 0.0, 0.98)
    max_round_frames = 1800       # 90 s at 20 Hz
    player_camera = {"pos": (1.25, -0.55, 1.30), "target": (0.48, 0.0, 0.86), "fovy": 50}

    # ----- scene -----
    def objects_xml(self):
        bodies = []
        for i, (name, half, rgba) in enumerate(BLOCKS):
            bodies.append(
                f'<body name="{name}" pos="0.5 {0.08 * i - 0.16} {TABLE_Z + half}"><freejoint/>'
                f'<geom type="box" size="{half} {half} {half}" rgba="{rgba}" mass="0.05" '
                f'friction="1.8 0.02 0.001" condim="4" solref="0.004 1"/></body>')
        # The base pad is a visual-only marker (policy cameras see it), moved per seed via mocap.
        p = PAD_HALF
        pad = f'''
        <body name="base_pad" mocap="true" pos="0.5 0 {TABLE_Z}">
          <geom type="box" size="{p} {p} 0.0008" pos="0 0 0.0008" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
          <geom type="box" size="{p - 0.006} {p - 0.006} 0.0009" pos="0 0 0.0009" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
          <geom type="box" size="0.012 0.012 0.0010" pos="0 0 0.0010" rgba="0.95 0.45 0.10 1" contype="0" conaffinity="0"/>
        </body>'''
        return "\n".join(bodies) + pad

    def decor_xml(self):
        return f'''
        <body name="wind_arrow" mocap="true" pos="0 0 -1">
          <geom type="capsule" fromto="-0.12 0 0 0.05 0 0" size="0.008" rgba="0.55 0.85 1 0.85"/>
          <geom type="box" pos="0.065 0 0" size="0.02 0.022 0.006" euler="0 0 0.785" rgba="0.55 0.85 1 0.85"/>
          <geom type="capsule" fromto="-0.10 0.05 0.03 0.0 0.05 0.03" size="0.005" rgba="0.75 0.92 1 0.6"/>
          <geom type="capsule" fromto="-0.10 -0.05 -0.03 0.0 -0.05 -0.03" size="0.005" rgba="0.75 0.92 1 0.6"/>
        </body>
        <body name="record_ring" mocap="true" pos="0 0 -1">
          <geom type="cylinder" size="0.055 0.0015" rgba="1 0.82 0.15 0.55"/>
        </body>
        <body name="next_ghost" mocap="true" pos="0 0 -1">
          <geom type="box" size="0.026 0.026 0.0015" rgba="0.3 1 0.6 0.35"/>
        </body>
        <geom type="box" pos="1.75 0 1.15" size="0.04 1.4 0.45" rgba="0.10 0.16 0.24 1"/>
        <geom type="box" pos="1.70 0 1.52" size="0.01 1.2 0.025" rgba="0.45 0.85 1 1"/>
        <geom type="box" pos="1.70 0 0.80" size="0.01 1.2 0.02" rgba="1 0.82 0.15 1"/>
        <geom type="cylinder" pos="1.70 -0.9 1.15" size="0.12 0.01" euler="0 1.5708 0" rgba="0.55 0.85 1 1"/>
        <geom type="cylinder" pos="1.70 0.9 1.15" size="0.12 0.01" euler="0 1.5708 0" rgba="0.55 0.85 1 1"/>
        '''

    def object_bodies(self):
        return [name for name, _, _ in BLOCKS] + ["base_pad"]

    # ----- helpers -----
    def _mocap(self, body):
        return self.model.body_mocapid[self.body_id(body)]

    def _move_mocap(self, body, pos, yaw=0.0):
        index = self._mocap(body)
        self.data.mocap_pos[index] = pos
        self.data.mocap_quat[index] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def face_up(self, name):
        """Cosine between world +z and the cube face normal closest to it (cubes have no 'top')."""
        return float(np.max(np.abs(self.data.xmat[self.body_id(name)].reshape(3, 3)[2, :])))

    def block_yaw(self, name):
        rot = self.data.xmat[self.body_id(name)].reshape(3, 3)
        # The most horizontal body axis gives the cube's yaw (mod 90 degrees).
        axis = rot[:, int(np.argmin(np.abs(rot[2, :])))]
        return wrap_quarter(math.atan2(axis[1], axis[0]))

    def block_speed(self, name):
        joint = self.model.body_jntadr[self.body_id(name)]
        dof = self.model.jnt_dofadr[joint]
        return float(np.linalg.norm(self.data.qvel[dof:dof + 3]))

    def top_z(self, tower=None):
        tower = self.tower if tower is None else tower
        return TABLE_Z + sum(2 * HALF[name] for name in tower)

    def expected_z(self, index):
        return TABLE_Z + sum(2 * HALF[name] for name in self.tower[:index]) + HALF[self.tower[index]]

    def offset(self, name):
        return float(np.linalg.norm(self.body_pos(name)[:2] - self.axis))

    def standing_count(self):
        """How many tower blocks (from the bottom) are still where they were stacked."""
        for index, name in enumerate(self.tower):
            pos = self.body_pos(name)
            if (abs(pos[2] - self.expected_z(index)) > TOWER_Z_TOL or self.offset(name) > TOWER_XY_TOL
                    or self.face_up(name) < TOWER_UPRIGHT):
                return index
        return len(self.tower)

    def free_blocks(self):
        return [name for name, _, _ in BLOCKS if name not in self.tower]

    def held_block(self):
        for name in self.free_blocks():
            if self.grasped(name):
                return name
        return None

    # ----- rules -----
    def reset(self):
        pad = np.array([self.rng.uniform(0.44, 0.56), self.rng.uniform(-0.14, 0.14)])
        self.axis = pad
        self._move_mocap("base_pad", (*pad, TABLE_Z))
        spots = []
        for name, half, _ in BLOCKS:
            for _ in range(200):
                xy = np.array([self.rng.uniform(0.32, 0.64), self.rng.uniform(-0.28, 0.28)])
                if np.linalg.norm(xy - pad) > 0.11 and all(np.linalg.norm(xy - s) > 0.085 for s in spots):
                    break
            spots.append(xy)
            yaw = self.rng.uniform(-math.pi / 4, math.pi / 4)
            self.set_free_body(name, (*xy, TABLE_Z + half + 0.001), (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        self._move_mocap("wind_arrow", (0, 0, -1))
        self.tower = []
        self.record = 0
        self.placed = 0
        self.gusts_survived = 0
        self.last_wind = "—"
        self.was_closed = False
        self.closure_block = None
        self.placing = None
        self.wind = None
        self.phase = "stack"
        self.label = STACK
        self.autopilot_plan = None
        self.message = "Stack a block on the pad!"
        self._update_markers()

    def _update_markers(self):
        top = self.top_z()
        self._move_mocap("next_ghost", (*self.axis, top + 0.0015) if self.phase == "stack" else (0, 0, -1))
        if self.record:
            self._move_mocap("record_ring", (*self.axis, TABLE_Z + self.record * 0.048))
        else:
            self._move_mocap("record_ring", (0, 0, -1))

    def before_action(self, command, buttons):
        if self.phase == "wind":
            # The arm is parked while the game blows: a scripted zero command, gripper open.
            hold = np.zeros(7)
            hold[6] = -1.0
            return StepControl(command=hold, source="game_script")
        return StepControl(command=np.asarray(command, float))

    def substep(self):
        if self.phase != "wind" or self.wind["frame"] >= GUST_FRAMES:
            return False
        profile = math.sin(math.pi * (self.wind["frame"] + 0.5) / GUST_FRAMES)
        count = len(self.tower)
        for index, name in enumerate(self.tower):
            share = (index + 1) / count          # the top takes the full gust, lower blocks less
            self.data.xfrc_applied[self.body_id(name), :2] = self.wind["dir"] * self.wind["force"] * profile * share
        return True

    def after_action(self):
        if self.round_over:
            return
        if self.frame >= self.max_round_frames:
            self.phase = "done"
            self.end_round("max_frames", f"Time's up — tower record {self.record}")
            return
        if self.phase == "stack":
            self._stack_step()
        elif self.phase == "wind":
            self._wind_step()
        self._update_markers()

    def _stack_step(self):
        closed = self.robot.grip > 0.5
        if closed and not self.was_closed:
            self.closure_block = None
            self.placing = None              # re-grasp cancels a pending placement check
        if closed:
            held = self.held_block()
            if held:
                self.closure_block = held
        standing = self.standing_count()
        if standing < len(self.tower):
            knocked = len(self.tower) - standing
            self.tower = self.tower[:standing]
            self.placing = None
            self.finish_chunk(False, reason="tower_knocked", blocks_lost=knocked, tower_height=standing)
            self.message = "Tower knocked over — rebuild it!"
        elif self.was_closed and not closed:
            if self.closure_block is None:
                self.finish_chunk(False, reason="released_without_block")
            else:
                self.placing = {"block": self.closure_block, "frames": 0, "stable": 0, "rest": 0}
        elif self.placing is not None:
            self._check_placement()
        self.was_closed = closed

    def _check_placement(self):
        placing = self.placing
        name = placing["block"]
        placing["frames"] += 1
        pos = self.body_pos(name)
        expected = self.top_z() + HALF[name]
        offset = self.offset(name)
        upright = self.face_up(name)
        height_error = pos[2] - expected
        left, right = self.finger_contacts(name)
        resting = self.block_speed(name) < SETTLE_SPEED
        on_support = abs(height_error) < HEIGHT_TOL_M and (
            not self.tower or self.touching(name, self.tower[-1]))
        placed_well = on_support and offset < MAX_OFFSET_M and upright > MIN_UPRIGHT
        clear = not (left or right) and self.robot.eef_pos()[2] > pos[2] + HALF[name] + CLEARANCE_M
        placing["stable"] = placing["stable"] + 1 if (resting and placed_well and clear) else 0
        placing["rest"] = placing["rest"] + 1 if (resting and not (left or right)) else 0
        evidence = dict(block=name, offset_m=round(offset, 4), height_m=round(float(pos[2] - TABLE_Z), 4),
                        height_error_m=round(float(height_error), 4), upright=round(upright, 4),
                        tower_height=len(self.tower) + 1)
        if placing["stable"] / 20 >= STABLE_S:
            self.tower.append(name)
            self.placed += 1
            self.score += 10 * len(self.tower)
            self.record = max(self.record, len(self.tower))
            self.placing = None
            self.finish_chunk(True, stable_s=round(placing["stable"] / 20, 3), left_contact=left,
                              right_contact=right, **evidence)
            self._start_wind()
        elif placing["rest"] / 20 >= STABLE_S and not placed_well:
            self.placing = None
            self.finish_chunk(False, reason="bad_placement" if on_support else "missed_tower", **evidence)
            self.message = "Off target — pick it up and try again." if on_support else "Missed the tower!"
        elif placing["frames"] / 20 >= PLACE_TIMEOUT_S:
            self.placing = None
            self.finish_chunk(False, reason="placement_timeout", **evidence)
            self.message = "Too slow — let go and move the gripper away."

    def _start_wind(self):
        angle = self.rng.uniform(-math.pi, math.pi)
        direction = np.array([math.cos(angle), math.sin(angle)])
        force = self.rng.uniform(*WIND_FORCE_N)
        self.wind = {"frame": 0, "dir": direction, "force": force, "angle": angle}
        self.phase = "wind"
        self.label = WIND
        self.message = f"Block {len(self.tower)} placed! Wind incoming..."
        top = self.top_z()
        self._move_mocap("wind_arrow", (*(self.axis - direction * 0.16), top - 0.03), angle)

    def _wind_step(self):
        wind = self.wind
        wind["frame"] += 1
        if wind["frame"] == GUST_FRAMES:
            self._move_mocap("wind_arrow", (0, 0, -1))
        if wind["frame"] < WIND_MIN_FRAMES:
            return
        settled = all(self.block_speed(name) < 0.05 for name, _, _ in BLOCKS)
        if not settled and wind["frame"] < WIND_MAX_FRAMES:
            return
        standing = self.standing_count()
        height = len(self.tower)
        survived = standing == height
        evidence = dict(force_n=round(wind["force"], 3), direction_deg=round(math.degrees(wind["angle"]), 1),
                        tower_height=height, standing_after=standing, survived=survived)
        self.wind = None
        self.tower = self.tower[:standing]
        if survived:
            self.gusts_survived += 1
            bonus = 15 * height
            self.score += bonus
            self.last_wind = f"Survived +{bonus}"
            self.finish_chunk(True, bonus=bonus, **evidence)
            if height >= GOAL_HEIGHT:
                self.phase = "done"
                self.end_round("round_complete", f"Tower of {height} stood firm! {int(self.score)} points")
                return
            self.message = f"Tower held! +{bonus}. Stack the next block."
        else:
            self.last_wind = f"Collapsed ({height}→{standing})"
            self.finish_chunk(False, **evidence)
            self.message = "The wind knocked it down — rebuild!"
        self.phase = "stack"
        self.label = STACK
        self.was_closed = self.robot.grip > 0.5
        self.closure_block = None
        self.placing = None

    def hud(self):
        return {"Height": f"{len(self.tower)}/{GOAL_HEIGHT}", "Record": str(self.record),
                "Wind": self.last_wind if self.phase != "wind" else "Blowing!",
                "Time": f"{max(0, self.max_round_frames - self.frame) / 20:.0f}s"}

    # ----- scripted expert, same command interface as a human -----
    def autopilot(self):
        robot = self.robot
        command = np.zeros(7)
        if self.phase != "stack":
            command[6] = -1
            self.autopilot_plan = None
            return command, {}
        plan = self.autopilot_plan
        if plan is None:
            plan = self.autopilot_plan = {"step": "choose", "wait": 0, "block": None, "fails": {}}
        if plan["step"] != "choose" and plan["block"] in self.tower:
            plan["step"] = "choose"
        plan["wait"] += 1
        step = plan["step"]
        eef = robot.eef_pos()

        if step == "choose":
            command[6] = -1
            candidates = []
            for name in self.free_blocks():
                pos = self.body_pos(name)
                if (pos[2] < TABLE_Z + HALF[name] + 0.01 and self.face_up(name) > 0.95
                        and np.linalg.norm(pos[:2] - self.axis) > 0.045 and plan["fails"].get(name, 0) < 3):
                    candidates.append(name)
            if not candidates:
                candidates = [n for n in self.free_blocks() if plan["fails"].get(n, 0) < 6]
            if not candidates:
                command, _ = self.steer_to(robot.target + [0, 0, 0.01])
                command[6] = -1
                return command, {}
            block = min(candidates, key=lambda n: np.linalg.norm(self.body_pos(n)[:2] - eef[:2]))
            plan.update(step="rise", block=block, wait=0)
            step = "rise"

        block = plan["block"]
        pos = self.body_pos(block)
        half = HALF[block]
        carry_z = max(self.top_z() + half + 0.05, TABLE_Z + 0.12)

        def advance(name):
            plan["step"], plan["wait"] = name, 0

        def give_up():
            plan["fails"][block] = plan["fails"].get(block, 0) + 1
            advance("choose")

        if step == "rise":
            command, reached = self.steer_to([eef[0], eef[1], max(carry_z, pos[2] + 0.10)], tolerance=0.02)
            command[6] = -1
            if reached or eef[2] > carry_z - 0.01 or plan["wait"] > 60:
                advance("above")
        elif step == "above":
            yaw = robot.euler[2] + wrap_quarter(self.block_yaw(block) - robot.euler[2])
            command, reached = self.steer_to(pos + [0, 0, 0.09], yaw=yaw, tolerance=0.008)
            command[6] = -1
            if reached and abs(wrap_pi(yaw - robot.euler[2])) < 0.04:
                advance("descend")
            elif plan["wait"] > 150:
                give_up()
        elif step == "descend":
            yaw = robot.euler[2] + wrap_quarter(self.block_yaw(block) - robot.euler[2])
            command, reached = self.steer_to(pos + [0, 0, 0.002], yaw=yaw, tolerance=0.008)
            command[6] = -1
            if reached:
                advance("close")
            elif plan["wait"] > 100:
                give_up()
        elif step == "close":
            command[6] = 1
            if plan["wait"] > 8:
                advance("lift")
        elif step == "lift":
            command, reached = self.steer_to([eef[0], eef[1], carry_z], tolerance=0.015)
            command[:2] = 0
            command[6] = 1
            if not self.grasped(block) and plan["wait"] > 5:
                command[6] = -1
                give_up()
            elif reached or eef[2] > carry_z - 0.01:
                advance("carry")
        elif step in ("carry", "lower"):
            in_hand = pos[:2] - eef[:2]
            goal_xy = self.axis - in_hand
            top_yaw = self.block_yaw(self.tower[-1]) if self.tower else 0.0
            yaw = robot.euler[2] + wrap_quarter(top_yaw - self.block_yaw(block))
            if not self.grasped(block):
                command[6] = -1
                give_up()
            elif step == "carry":
                command, reached = self.steer_to([*goal_xy, carry_z], yaw=yaw, tolerance=0.006)
                command[6] = 1
                xy_error = np.linalg.norm(pos[:2] - self.axis)
                if reached and xy_error < 0.004 and abs(wrap_pi(yaw - robot.euler[2])) < 0.03:
                    advance("lower")
                elif plan["wait"] > 200:
                    advance("lower")
            else:
                expected = self.top_z() + half
                goal_z = eef[2] - (pos[2] - (expected + 0.003))
                gain = 0.5 if pos[2] - expected > 0.02 else 0.3
                command, _ = self.steer_to([*goal_xy, goal_z], gain=gain, yaw=yaw)
                command[6] = 1
                if (pos[2] - expected < 0.005 and np.linalg.norm(pos[:2] - self.axis) < 0.006) or plan["wait"] > 120:
                    advance("release")
        elif step == "release":
            command[6] = -1
            if plan["wait"] > 6:
                advance("retreat")
        elif step == "retreat":
            command, _ = self.steer_to([eef[0], eef[1], pos[2] + half + 0.07], tolerance=0.01)
            command[:2] = 0
            command[6] = -1
            if self.placing is None and plan["wait"] > 3:
                # The placement was judged (and failed) while we are still in the stack phase.
                advance("choose")
            elif plan["wait"] > 140:
                advance("choose")
        if step not in ("close", "release"):
            command[:3] += self.rng.normal(0, 0.03, 3)
        return np.clip(command, -1, 1), {}
