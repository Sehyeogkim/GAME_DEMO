"""Color Sort Rush: put each block in the bin of its color (trainable), with combo, timer and wave bonuses (game fun)."""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label, StepControl

COLORS = {
    "red": (0.90, 0.18, 0.15),
    "green": (0.15, 0.72, 0.25),
    "blue": (0.15, 0.35, 0.92),
}
BLOCK_HALF = 0.022
BLOCKS = [(f"{color}_block_{i}", color) for color in COLORS for i in (1, 2)]
BLOCK_COLOR = dict(BLOCKS)
# Open-top bins along the far edge of the reachable area. Inner half-size, wall height.
BIN_X = 0.645
BIN_Y = {"red": -0.19, "green": 0.0, "blue": 0.19}
BIN_HALF = 0.068
WALL_TOP = 0.80
PLAY_X = (0.29, 0.50)
PLAY_Y = (-0.27, 0.27)
MIN_GAP = 0.085

SORT = Label("sort_block", "Put a block in the bin of the same color.", True)
WAVE = Label("wave_bonus", "Wave bonus: the arm returns home while the bins flash.", False, "game_fun")

REST_S = 0.3                 # block must rest in its bin this long
REST_SPEED = 0.05            # m/s, "resting"
SETTLE_TIMEOUT_S = 2.0       # after a release, how long to wait for the block to come to rest
ROUND_S = 90.0
WAVE_FRAMES = 22
WAVE_EVERY = 3
BASE_POINTS = 100
COMBO_WINDOW_S = 12.0        # a sort slower than this breaks the combo


class ColorSort(ArcadeGame):
    name = "color_sort"
    title = "Color Sort Rush"
    version = "1"
    port = 8803
    tagline = "Sort the blocks into matching bins before time runs out. Chain clean sorts for combos."
    skills = {"sort_block": "Pick one block and place it so it rests inside the bin of its own color, "
                            "free of the fingers, for 0.3 s."}
    phases = ["sort", "wave", "done"]
    controls = [("Goal", "Drop each block into the bin of the same color"),
                ("Combo", "Clean, fast sorts raise the combo; a miss resets it")]
    home_position = (0.40, 0.0, 0.98)
    max_round_frames = int(ROUND_S * 20) + 40

    # ----- scene -----
    def objects_xml(self):
        parts = []
        for name, color in BLOCKS:
            r, g, b = COLORS[color]
            parts.append(
                f'<body name="{name}" pos="0.4 0 {0.74 + BLOCK_HALF}"><freejoint/>'
                f'<geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" rgba="{r} {g} {b} 1" mass="0.05" '
                f'friction="1.8 0.02 0.001" condim="4" solref="0.004 1"/></body>')
        h, t = BIN_HALF, 0.005
        wall_half_z = (WALL_TOP - 0.74) / 2
        wall_z = 0.74 + wall_half_z
        for color, y in BIN_Y.items():
            r, g, b = COLORS[color]
            x = BIN_X
            # Colored floor (visible to the policy) plus walls tinted with the bin color.
            parts.append(f'''
            <geom name="bin_{color}_floor" type="box" pos="{x} {y} 0.7415" size="{h} {h} 0.0015" rgba="{r} {g} {b} 1"/>
            <geom type="box" pos="{x + h + t} {y} {wall_z}" size="{t} {h + 2 * t} {wall_half_z}" rgba="{r * 0.7} {g * 0.7} {b * 0.7} 1"/>
            <geom type="box" pos="{x - h - t} {y} {wall_z}" size="{t} {h + 2 * t} {wall_half_z}" rgba="{r * 0.7} {g * 0.7} {b * 0.7} 1"/>
            <geom type="box" pos="{x} {y + h + t} {wall_z}" size="{h} {t} {wall_half_z}" rgba="{r * 0.7} {g * 0.7} {b * 0.7} 1"/>
            <geom type="box" pos="{x} {y - h - t} {wall_z}" size="{h} {t} {wall_half_z}" rgba="{r * 0.7} {g * 0.7} {b * 0.7} 1"/>
            ''')
        return "\n".join(parts)

    def decor_xml(self):
        glows = []
        for color, y in BIN_Y.items():
            r, g, b = COLORS[color]
            glows.append(f'<geom name="glow_{color}" type="box" pos="{BIN_X} {y} {WALL_TOP + 0.004}" '
                         f'size="{BIN_HALF + 0.012} {BIN_HALF + 0.012} 0.003" rgba="{r} {g} {b} 0.25"/>')
        return "\n".join(glows) + '''
        <geom type="box" pos="1.3 0 1.1" size="0.04 1.2 0.5" rgba="0.10 0.10 0.16 1"/>
        <geom type="box" pos="1.26 -0.4 1.45" size="0.01 0.19 0.03" rgba="0.90 0.18 0.15 1"/>
        <geom type="box" pos="1.26 0 1.45" size="0.01 0.19 0.03" rgba="0.15 0.72 0.25 1"/>
        <geom type="box" pos="1.26 0.4 1.45" size="0.01 0.19 0.03" rgba="0.15 0.35 0.92 1"/>
        <geom type="box" pos="1.26 0 0.82" size="0.01 1.0 0.02" rgba="1 0.85 0.2 1"/>
        '''

    def object_bodies(self):
        return [name for name, _ in BLOCKS]

    # ----- rules -----
    def reset(self):
        spots = []
        for name, _ in BLOCKS:
            for _ in range(200):
                xy = np.array([self.rng.uniform(*PLAY_X), self.rng.uniform(*PLAY_Y)])
                if all(np.linalg.norm(xy - s) > MIN_GAP for s in spots):
                    break
            spots.append(xy)
            yaw = self.rng.uniform(-math.pi / 4, math.pi / 4)
            self.set_free_body(name, (*xy, 0.74 + BLOCK_HALF + 0.002), (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        self.glow_ids = {color: self.geom_id(f"glow_{color}") for color in COLORS}
        self.glow_rgba = {color: self.model.geom_rgba[i].copy() for color, i in self.glow_ids.items()}
        self.phase = "sort"
        self.label = SORT
        self.sorted = set()
        self.combo = 1
        self.best_combo = 1
        self.multiplier = 1.0
        self.rest_frames = {name: 0 for name, _ in BLOCKS}
        self.wrong_flagged = set()     # blocks already penalised for sitting in a wrong bin
        self.released = None           # block let go by the fingers, waiting for it to settle
        self.released_frames = 0
        self.held_last = None
        self.was_closed = False
        self.chunk_start = 0
        self.wave_frames = 0
        self.waves = 0
        self.autopilot_plan = None
        self.message = "Sort the blocks by color!"

    # --- physical queries ---
    def block_speed(self, name):
        body = self.body_id(name)
        joint = self.model.body_jntadr[body]
        dadr = self.model.jnt_dofadr[joint]
        return float(np.linalg.norm(self.data.qvel[dadr:dadr + 3]))

    def bin_of(self, name):
        """Color of the bin whose interior holds the block (its bottom below the wall top), else None."""
        pos = self.body_pos(name)
        if pos[2] - BLOCK_HALF > WALL_TOP:
            return None
        for color, y in BIN_Y.items():
            if abs(pos[0] - BIN_X) < BIN_HALF and abs(pos[1] - y) < BIN_HALF:
                return color
        return None

    def held_block(self):
        for name, _ in BLOCKS:
            if name not in self.sorted and self.grasped(name):
                return name
        return None

    def time_left(self):
        return max(0.0, ROUND_S - self.frame / 20)

    # --- per-step rules ---
    def before_action(self, command, buttons):
        if self.phase == "wave":
            # The game drives the arm home; this is a game_fun chunk, never training data.
            target = np.array(self.home_position)
            error = target - self.robot.target
            scripted = np.zeros(7)
            scripted[:3] = np.clip(error / 0.012 * 0.5, -1, 1)
            scripted[6] = -1
            return StepControl(command=scripted, source="game_script")
        return StepControl(command=np.asarray(command, float))

    def after_action(self):
        if self.round_over:
            return
        if self.phase == "wave":
            self._wave_step()
        elif self.phase == "sort":
            self._sort_step()
        if not self.round_over and self.time_left() <= 0:
            self.phase = "done"
            self.end_round("max_frames", f"Time's up! {len(self.sorted)}/6 sorted, {int(self.score)} points")

    def _sort_step(self):
        closed = self.robot.grip > 0.5
        held = self.held_block()
        if held:
            self.held_last = held
            self.released = None
        elif self.was_closed and not closed:
            if self.held_last is not None:
                self.released = self.held_last
                self.released_frames = 0
            else:
                # Gripper opened again without ever holding a block: missed grasp.
                self._fail("released_without_block")
            self.held_last = None
        elif not closed and self.held_last is not None and self.released is None:
            # Block slipped out of a closed-then-opened grasp without an observed release edge.
            self.released, self.released_frames, self.held_last = self.held_last, 0, None
        self.was_closed = closed

        # Judge every unsorted block that is at rest inside a bin.
        for name, color in BLOCKS:
            if name in self.sorted:
                continue
            in_bin = self.bin_of(name)
            left, right = self.finger_contacts(name)
            resting = in_bin is not None and not (left or right) and self.block_speed(name) < REST_SPEED
            self.rest_frames[name] = self.rest_frames[name] + 1 if resting else 0
            if in_bin is None:
                self.wrong_flagged.discard(name)
            if self.rest_frames[name] >= round(REST_S * 20):
                if in_bin == color:
                    self._success(name, color)
                    return
                if name not in self.wrong_flagged:
                    self.wrong_flagged.add(name)
                    if self.released == name:
                        self.released = None
                    self._fail("wrong_bin", block=name, block_color=color, bin=in_bin)
                    self.message = f"Wrong bin! That {color} block belongs in the {color} bin."
                    return

        if self.released is not None:
            self.released_frames += 1
            name = self.released
            pos = self.body_pos(name)
            settled = self.block_speed(name) < REST_SPEED and self.released_frames >= 6
            if (settled and self.bin_of(name) is None) or self.released_frames > SETTLE_TIMEOUT_S * 20:
                self.released = None
                self._fail("dropped_outside_bin", block=name, block_pos=[round(float(v), 3) for v in pos])
                self.message = "Dropped outside the bins — combo lost."

    def _success(self, name, color):
        pos = self.body_pos(name)
        seconds = (self.frame - self.chunk_start) / 20
        if seconds > COMBO_WINDOW_S:
            self.combo = 1
        points = int(BASE_POINTS * self.combo * self.multiplier + max(0.0, COMBO_WINDOW_S - seconds) * 10)
        self.score += points
        self.finish_chunk(True, block=name, block_color=color, bin=color,
                          block_pos=[round(float(v), 4) for v in pos],
                          rest_s=round(self.rest_frames[name] / 20, 3),
                          finger_contact=False, chunk_seconds=round(seconds, 2),
                          combo=self.combo, points=points)
        self.sorted.add(name)
        self.released = None if self.released == name else self.released
        self.held_last = None
        self.message = f"{color.title()} sorted! +{points}  (combo x{self.combo})"
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo - 1)
        self.chunk_start = self.frame
        if len(self.sorted) % WAVE_EVERY == 0:
            self.phase = "wave"
            self.label = WAVE
            self.wave_frames = 0
            self.waves += 1

    def _fail(self, reason, **evidence):
        self.finish_chunk(False, reason=reason, **evidence)
        self.combo = 1
        self.chunk_start = self.frame
        if reason == "released_without_block":
            self.message = "Missed the grab — combo reset."

    def _wave_step(self):
        self.wave_frames += 1
        on = (self.wave_frames // 3) % 2 == 0
        for color, gid in self.glow_ids.items():
            rgba = self.glow_rgba[color].copy()
            rgba[3] = 0.9 if on else 0.15
            self.model.geom_rgba[gid] = rgba
        if self.wave_frames == 1:
            gain = 0.5 if self.combo > WAVE_EVERY else 0.25   # clean streak through the wave → bigger boost
            self.multiplier += gain
            bonus = 50 * (self.combo - 1)
            self.score += bonus
            self.wave_bonus = (gain, bonus)
            self.message = f"WAVE {self.waves} BONUS! multiplier x{self.multiplier:.2f}  +{bonus}"
        if self.wave_frames >= WAVE_FRAMES:
            for color, gid in self.glow_ids.items():
                self.model.geom_rgba[gid] = self.glow_rgba[color]
            gain, bonus = self.wave_bonus
            self.finish_chunk(True, wave=self.waves, multiplier=self.multiplier, bonus=bonus)
            self.was_closed = self.robot.grip > 0.5
            self.held_last = None
            self.chunk_start = self.frame
            if len(self.sorted) == len(BLOCKS):
                self.phase = "done"
                self.end_round("round_complete", f"All sorted! {int(self.score)} points, best combo x{self.best_combo}")
            else:
                self.phase = "sort"
                self.label = SORT

    def hud(self):
        return {"Sorted": f"{len(self.sorted)}/{len(BLOCKS)}",
                "Combo": f"x{self.combo}",
                "Multiplier": f"x{self.multiplier:.2f}",
                "Time": f"{self.time_left():.0f}s"}

    def _free_drop_spot(self, block):
        """Spot in the block's bin farthest from blocks already there (along x, so fingers clear the walls)."""
        color = BLOCK_COLOR[block]
        centre = np.array([BIN_X, BIN_Y[color]])
        occupied = [self.body_pos(n)[:2] for n, _ in BLOCKS if n != block and self.bin_of(n) is not None]
        spots = [centre + [dx, 0] for dx in (-0.03, 0.03, 0.0)]
        if not occupied:
            return spots[0]
        return max(spots, key=lambda p: min(np.linalg.norm(p - o) for o in occupied))

    # ----- scripted expert, same command interface as a human -----
    def autopilot(self):
        robot = self.robot
        command = np.zeros(7)
        if self.phase != "sort":
            self.autopilot_plan = None
            return command, {}
        plan = self.autopilot_plan
        if plan is None or plan["block"] in self.sorted:
            candidates = [n for n, _ in BLOCKS if n not in self.sorted and self.bin_of(n) is None]
            if not candidates:
                candidates = [n for n, _ in BLOCKS if n not in self.sorted]
            if not candidates:
                return command, {}
            eef = robot.eef_pos()
            block = min(candidates, key=lambda n: np.linalg.norm(self.body_pos(n)[:2] - eef[:2]))
            plan = self.autopilot_plan = {"block": block, "step": "above", "wait": 0, "tries": 0}
        block = plan["block"]
        pos = self.body_pos(block)
        bin_xy = plan.get("drop")
        if bin_xy is None:
            bin_xy = plan["drop"] = self._free_drop_spot(block)
        rot = self.data.xmat[self.body_id(block)].reshape(3, 3)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4   # cube: any face pair works
        step = plan["step"]
        if step == "above":
            command, reached = self.steer_to(pos + [0, 0, 0.10], yaw=yaw, tolerance=0.012)
            command[6] = -1
            if reached and abs(((yaw - robot.euler[2]) + math.pi) % (2 * math.pi) - math.pi) < 0.05:
                plan["step"] = "descend"
        elif step == "descend":
            command, reached = self.steer_to(pos + [0, 0, 0.002], yaw=yaw, tolerance=0.012)
            command[6] = -1
            if reached:
                plan["step"], plan["wait"] = "close", 0
        elif step == "close":
            command[6] = 1
            plan["wait"] += 1
            if plan["wait"] > 8:
                plan["step"] = "lift"
        elif step == "lift":
            command, reached = self.steer_to([robot.target[0], robot.target[1], 0.90], tolerance=0.015)
            command[6] = 1
            if not self.grasped(block):
                plan["step"] = "retry"
            elif reached:
                plan["step"] = "carry"
        elif step == "carry":
            command, reached = self.steer_to([*bin_xy, 0.90], tolerance=0.015)
            command[6] = 1
            if not self.grasped(block):
                plan["step"] = "retry"
            elif reached:
                plan["step"] = "lower"
        elif step == "lower":
            command, reached = self.steer_to([*bin_xy, 0.83], tolerance=0.012)
            command[6] = 1
            if reached:
                plan["step"], plan["wait"] = "release", 0
        elif step == "release":
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 6:
                plan["step"], plan["wait"] = "clear", 0
        elif step == "clear":
            command, _ = self.steer_to([*bin_xy, 0.93], tolerance=0.02)
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 40:   # not sorted yet after a long wait: try that block again
                plan["step"], plan["drop"] = "above", None
        elif step == "retry":
            # Open and rise before going back above the block.
            command, reached = self.steer_to([robot.target[0], robot.target[1], 0.90], tolerance=0.02)
            command[6] = -1
            plan["tries"] += 1
            if reached or plan["tries"] > 60:
                plan["step"], plan["tries"] = "above", 0
        command[:3] += self.rng.normal(0, 0.03, 3)
        return np.clip(command, -1, 1), {}
