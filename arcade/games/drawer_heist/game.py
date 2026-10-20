"""Drawer Heist: pull a drawer open by its handle and pick the gem (trainable), then stash it in the vault (game fun).

A low cabinet holds two drawers side by side. Each drawer is a body on a slide joint that opens toward
the robot (-x). Exactly one drawer hides a gem. Opening the empty one is still a real, successful
open_drawer chunk; the game then says "Empty!" and the player opens the other one.
"""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label

TABLE_Z = 0.74
CABINET_FRONT_X = 0.56            # drawer fronts sit flush with this plane when closed
CABINET_DEPTH = 0.26
CABINET_TOP_Z = 0.828             # underside of the cabinet top (0.088 m above the table; top surface 0.10 m)
DRAWERS = {"drawer_left": 0.10, "drawer_right": -0.10}   # body name -> centre y
DRAWER_TRAVEL = 0.15
OPEN_M = 0.10                     # slide qpos that counts as open
HANDLE_STANDOFF = 0.06            # T-handle: a pull bar sticking out of the drawer face along x, knob at its end
GRASP_FROM_KNOB = 0.022           # where fingers pinch the bar, measured back from the knob
HANDLE_Z_LOCAL = 0.072            # bar height above the table
FLOOR_TOP_Z = TABLE_Z + 0.016
GEM_HALF = 0.014
GEM_REST_Z = FLOOR_TOP_Z + GEM_HALF
VAULT = np.array([0.36, -0.33])
VAULT_HALF = 0.07
LIFT_M = 0.06
HOLD_S = 0.3
ROUND_S = 90
CONTROL_HZ = 20

OPEN = Label("open_drawer", "Grasp a drawer handle, pull the drawer open, and let go.", True)
PICK = Label("pick_gem", "Pick the gem out of the open drawer and lift it.", True)
CARRY = Label("stash_gem", "Carry the gem to the vault and drop it.", False, "game_fun")
# Gem rarity roll: name, value, probability.
RARITY = [("Quartz", 20, 0.55), ("Sapphire", 50, 0.28), ("Ruby", 100, 0.13), ("Diamond", 250, 0.04)]
OBJECT_PHYSICS = 'friction="1.8 0.02 0.001" condim="4" solref="0.004 1"'


def _yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class DrawerHeist(ArcadeGame):
    name = "drawer_heist"
    title = "Drawer Heist"
    version = "1"
    port = 8807
    tagline = "Crack the right drawer, snatch the gem, stash it in the vault."
    skills = {
        "open_drawer": "Grasp a drawer handle with both fingers, pull the drawer open >= 0.10 m "
                       "(slide joint position), then release the handle.",
        "pick_gem": "Pick the gem from the open drawer and lift it 6 cm above where it rested, "
                    "held with both fingers for 0.3 s.",
    }
    phases = ["open", "pick", "carry", "done"]
    controls = [("Goal", "Pull a drawer open by its handle, grab the gem, drop it in the vault"),
                ("Tip", "Pinch the handle bar just behind its gold knob, then pull")]
    home_position = (0.40, 0.0, 0.98)
    max_round_frames = ROUND_S * CONTROL_HZ
    player_camera = {"pos": (-0.30, -0.72, 1.55), "target": (0.50, 0.0, 0.80), "fovy": 52}

    # ----- scene -----
    def objects_xml(self):
        x0, depth = CABINET_FRONT_X, CABINET_DEPTH
        cx = x0 + depth / 2
        wall_half_z = (CABINET_TOP_Z - TABLE_Z) / 2
        wall_z = TABLE_Z + wall_half_z
        wood = 'rgba="0.42 0.30 0.22 1"'
        cabinet = f'''
        <geom name="cabinet_top" type="box" pos="{cx} 0 {CABINET_TOP_Z + 0.006}" size="{depth / 2} 0.205 0.006" {wood}/>
        <geom name="cabinet_back" type="box" pos="{x0 + depth - 0.004} 0 {wall_z}" size="0.004 0.205 {wall_half_z}" {wood}/>
        <geom name="cabinet_side_l" type="box" pos="{cx} 0.197 {wall_z}" size="{depth / 2} 0.008 {wall_half_z}" {wood}/>
        <geom name="cabinet_side_r" type="box" pos="{cx} -0.197 {wall_z}" size="{depth / 2} 0.008 {wall_half_z}" {wood}/>
        <geom name="cabinet_divider" type="box" pos="{cx} 0 {wall_z}" size="{depth / 2} 0.006 {wall_half_z}" {wood}/>
        '''
        drawers = []
        for name, y in DRAWERS.items():
            panel = "0.78 0.62 0.36 1" if y > 0 else "0.36 0.62 0.78 1"
            hx = -HANDLE_STANDOFF
            drawers.append(f'''
        <body name="{name}" pos="{x0} {y} {TABLE_Z}">
          <joint name="{name}_slide" type="slide" axis="-1 0 0" range="0 {DRAWER_TRAVEL}" limited="true"
                 damping="4" frictionloss="0.4" armature="0.05"/>
          <geom name="{name}_panel" type="box" pos="0.006 0 0.044" size="0.006 0.083 0.037" mass="0.12" rgba="{panel}"/>
          <geom name="{name}_floor" type="box" pos="0.106 0 0.012" size="0.094 0.077 0.004" mass="0.10" rgba="0.55 0.45 0.35 1"/>
          <geom name="{name}_wall_l" type="box" pos="0.106 0.081 0.036" size="0.094 0.004 0.028" mass="0.04" rgba="0.55 0.45 0.35 1"/>
          <geom name="{name}_wall_r" type="box" pos="0.106 -0.081 0.036" size="0.094 0.004 0.028" mass="0.04" rgba="0.55 0.45 0.35 1"/>
          <geom name="{name}_wall_b" type="box" pos="0.196 0 0.036" size="0.004 0.077 0.028" mass="0.04" rgba="0.55 0.45 0.35 1"/>
          <geom name="{name}_handle" type="box" pos="{hx / 2} 0 {HANDLE_Z_LOCAL}" size="{HANDLE_STANDOFF / 2} 0.015 0.006" mass="0.02"
                rgba="0.85 0.85 0.88 1" {OBJECT_PHYSICS}/>
          <geom name="{name}_knob" type="box" pos="{hx - 0.006} 0 {HANDLE_Z_LOCAL}" size="0.006 0.034 0.008" mass="0.01"
                rgba="0.95 0.75 0.15 1" {OBJECT_PHYSICS}/>
        </body>''')
        gem = f'''
        <body name="gem" pos="{x0 + 0.06} 0.10 {GEM_REST_Z + 0.002}">
          <freejoint/>
          <geom name="gem" type="box" size="{GEM_HALF} {GEM_HALF} {GEM_HALF}" mass="0.05" rgba="0.65 0.20 0.95 1" {OBJECT_PHYSICS}/>
        </body>'''
        vx, vy, h = VAULT[0], VAULT[1], VAULT_HALF
        vault = f'''
        <geom name="vault_pad" type="box" pos="{vx} {vy} {TABLE_Z + 0.002}" size="{h} {h} 0.002" rgba="0.15 0.16 0.20 1"/>
        <geom name="vault_rim_x0" type="box" pos="{vx - h} {vy} {TABLE_Z + 0.012}" size="0.004 {h} 0.012" rgba="0.95 0.75 0.15 1"/>
        <geom name="vault_rim_x1" type="box" pos="{vx + h} {vy} {TABLE_Z + 0.012}" size="0.004 {h} 0.012" rgba="0.95 0.75 0.15 1"/>
        <geom name="vault_rim_y0" type="box" pos="{vx} {vy - h} {TABLE_Z + 0.012}" size="{h} 0.004 0.012" rgba="0.95 0.75 0.15 1"/>
        <geom name="vault_rim_y1" type="box" pos="{vx} {vy + h} {TABLE_Z + 0.012}" size="{h} 0.004 0.012" rgba="0.95 0.75 0.15 1"/>
        '''
        return cabinet + "".join(drawers) + gem + vault

    def decor_xml(self):
        vx, vy = VAULT
        return f'''
        <geom type="box" pos="1.3 0 1.1" size="0.04 1.2 0.5" rgba="0.08 0.07 0.12 1"/>
        <geom type="box" pos="1.26 0 1.45" size="0.01 1.0 0.025" rgba="0.95 0.75 0.15 1"/>
        <geom type="box" pos="1.26 0 0.80" size="0.01 1.0 0.02" rgba="0.9 0.15 0.2 1"/>
        <geom type="box" pos="{vx} {vy} {TABLE_Z + 0.03}" size="0.078 0.078 0.002" rgba="0.95 0.75 0.15 0.30"/>
        <geom type="cylinder" pos="{vx + 0.09} {vy - 0.09} {TABLE_Z + 0.10}" size="0.008 0.10" rgba="0.95 0.75 0.15 1"/>
        <geom type="box" pos="{vx + 0.09} {vy - 0.09} {TABLE_Z + 0.21}" size="0.04 0.004 0.025" rgba="0.95 0.75 0.15 1"/>
        <geom type="cylinder" fromto="{CABINET_FRONT_X - 0.02} -0.25 {TABLE_Z + 0.004} {CABINET_FRONT_X - 0.02} 0.25 {TABLE_Z + 0.004}" size="0.002" rgba="1 0.1 0.15 0.8"/>
        <geom type="sphere" pos="1.25 -0.7 1.2" size="0.06" rgba="0.9 0.15 0.2 1"/>
        <geom type="sphere" pos="1.25 0.7 1.2" size="0.06" rgba="0.95 0.75 0.15 1"/>
        '''

    def object_bodies(self):
        return [*DRAWERS, "gem"]

    # ----- state helpers -----
    def drawer_q(self, name):
        joint = self.model.body_jntadr[self.body_id(name)]
        return float(self.data.qpos[self.model.jnt_qposadr[joint]])

    def handle_pos(self, name):
        """Grasp point on the pull bar, just behind the knob."""
        knob = self.data.geom_xpos[self.geom_id(f"{name}_knob")]
        return knob + [GRASP_FROM_KNOB, 0.0, 0.0]

    def gem_in_vault(self):
        pos = self.body_pos("gem")
        left, right = self.finger_contacts("gem")
        return bool(np.all(np.abs(pos[:2] - VAULT) < VAULT_HALF) and pos[2] < TABLE_Z + 0.06
                    and not (left or right))

    # ----- rules -----
    def reset(self):
        self.gem_drawer = list(DRAWERS)[int(self.rng.integers(2))]
        local_x = self.rng.uniform(0.050, 0.065)
        local_y = self.rng.uniform(-0.02, 0.02)
        yaw = self.rng.uniform(-0.3, 0.3)
        self.set_free_body("gem", (CABINET_FRONT_X + local_x, DRAWERS[self.gem_drawer] + local_y, GEM_REST_Z + 0.001),
                           _yaw_quat(yaw))
        self.phase = "open"
        self.label = OPEN
        self.opened = []
        self.was_closed = False
        self.grabbed = None            # drawer whose handle was grasped during the current close
        self.release_pending = None    # drawer waiting for the fingers to leave the handle
        self.slip_frames = 0
        self.chunk_start = 0
        self.open_bonus = 0.0
        self.clean = True              # no slipped/failed grasp so far: keeps the heist bonus
        self.hold_s = 0.0
        self.gem_rest_z = GEM_REST_Z
        self.gem_prev = None
        self.still_frames = 0
        self.loot = None
        self.gem_status = "Hidden"
        self.autopilot_plan = None
        self.message = "Find the gem! Pull a drawer open."

    def _set_label(self, label):
        self.label = label
        self.chunk_start = self.frame
        self.hold_s = 0.0

    def after_action(self):
        closed = self.robot.grip > 0.5
        gem = self.body_pos("gem")
        speed = 0.0 if self.gem_prev is None else float(np.linalg.norm(gem - self.gem_prev)) * CONTROL_HZ
        self.gem_prev = gem
        if self.phase == "open":
            self._open_rules(closed)
        elif self.phase == "pick":
            self._pick_rules(closed, gem)
        elif self.phase == "carry":
            self._carry_rules(gem, speed)
        if not self.round_over and self.frame >= self.max_round_frames:
            self.phase = "done"
            self.end_round("max_frames", "Time's up! The guards are coming.")
        self.was_closed = closed

    def _open_rules(self, closed):
        if closed:
            for name in DRAWERS:
                if self.grasped(name):
                    self.grabbed = name
                    self.release_pending = None
            if self.grabbed is not None:
                # While pulling, the load sits on one finger; a slip means neither finger touches the handle.
                held = any(self.finger_contacts(self.grabbed))
                self.slip_frames = 0 if held else self.slip_frames + 1
                if self.slip_frames == 4 and self.drawer_q(self.grabbed) < OPEN_M:
                    self.clean = False
                    self.message = "Slipped off the handle!"
        elif self.was_closed:
            # Gripper just reopened: did that pull open a new drawer?
            drawer = self.grabbed
            if drawer is not None and drawer not in self.opened and self.drawer_q(drawer) >= OPEN_M:
                self.release_pending = drawer
            else:
                reason = ("no_handle_grasp" if drawer is None else
                          "already_open" if drawer in self.opened else "released_before_open")
                self.finish_chunk(False, reason=reason, drawer=drawer,
                                  slide_q=round(self.drawer_q(drawer), 4) if drawer else None)
                self._set_label(OPEN)
                if drawer is not None and drawer not in self.opened:
                    self.clean = False
                    self.message = "Not open enough. Pull it further!"
            self.grabbed = None
            self.slip_frames = 0
        drawer = self.release_pending
        if drawer is not None and not closed:
            left, right = self.finger_contacts(drawer)
            if not (left or right):
                seconds = (self.frame - self.chunk_start) / CONTROL_HZ
                quick = max(0.0, 25.0 - 2.0 * seconds)
                self.finish_chunk(True, drawer=drawer, slide_q=round(self.drawer_q(drawer), 4),
                                  released=True, seconds=round(seconds, 2))
                self.release_pending = None
                self.opened.append(drawer)
                self.open_bonus += quick if self.clean else 0.0
                if drawer == self.gem_drawer:
                    self.phase = "pick"
                    self._set_label(PICK)
                    self.gem_rest_z = float(self.body_pos("gem")[2])
                    self.gem_status = "Found"
                    self.message = "A gem! Grab it."
                else:
                    self._set_label(OPEN)
                    self.message = "Empty! Try the other drawer."

    def _pick_rules(self, closed, gem):
        held = self.grasped("gem")
        if not held:
            self.gem_rest_z = min(self.gem_rest_z, float(gem[2]))
        lift = float(gem[2]) - self.gem_rest_z
        if held and lift >= LIFT_M:
            self.hold_s += 1 / CONTROL_HZ
            if self.hold_s >= HOLD_S - 1e-9:
                left, right = self.finger_contacts("gem")
                self.finish_chunk(True, lift_m=round(lift, 4), hold_s=round(self.hold_s, 3),
                                  left_contact=left, right_contact=right)
                self.phase = "carry"
                self._set_label(CARRY)
                self.gem_status = "Held"
                self.message = "Got it! Stash it in the vault."
            return
        self.hold_s = 0.0
        if self.was_closed and not closed:
            self.finish_chunk(False, reason="released_before_stable_lift", lift_m=round(lift, 4))
            self._set_label(PICK)
            self.clean = False
            self.message = "Fumbled! Try again."
        elif not held and self.drawer_q(self.gem_drawer) < OPEN_M - 0.03 and gem[0] > CABINET_FRONT_X - 0.01:
            # The gem drawer was pushed shut with the gem inside: open it again.
            self.finish_chunk(False, reason="drawer_closed_on_gem")
            self.opened.remove(self.gem_drawer)
            self.phase = "open"
            self._set_label(OPEN)
            self.gem_status = "Hidden"
            self.message = "The drawer slid shut! Open it again."

    def _carry_rules(self, gem, speed):
        held = self.grasped("gem")
        self.still_frames = self.still_frames + 1 if (not held and speed < 0.03) else 0
        if self.gem_in_vault() and self.still_frames >= 3:
            name, value = self._roll_rarity()
            bonus = round(self.open_bonus) + 15 if self.clean else 0
            loot = value + bonus
            self.score += loot
            self.loot = {"rarity": name, "value": value, "bonus": bonus, "total": loot}
            self.finish_chunk(True, rarity=name, value=value, bonus=bonus, clean=self.clean)
            self.gem_status = "In vault"
            self.phase = "done"
            self.end_round("round_complete", f"{name}! Loot {loot} (bonus +{bonus})")
        elif self.still_frames >= 5:
            self.finish_chunk(False, reason="dropped_outside_vault", gem_pos=[round(float(v), 3) for v in gem])
            self.clean = False
            self.phase = "pick"
            self._set_label(PICK)
            self.gem_rest_z = float(gem[2])
            self.gem_status = "Dropped"
            self.message = "Dropped it! Bonus lost. Grab it again."

    def _roll_rarity(self):
        roll = self.rng.random()
        for name, value, p in RARITY:
            if roll < p:
                return name, value
            roll -= p
        return RARITY[0][:2]

    def hud(self):
        seconds_left = max(0, ROUND_S - self.frame / CONTROL_HZ)
        bonus = "lost" if not self.clean else f"+{round(self.open_bonus) + 15}"
        return {"Drawers": f"{len(self.opened)}/2 opened",
                "Gem": self.gem_status,
                "Loot": f"{self.loot['rarity']} {self.loot['total']}" if self.loot else "—",
                "Bonus": bonus,
                "Time": f"{seconds_left:.0f}s"}

    # ----- scripted expert, same command interface as a human -----
    def _new_plan(self):
        plan = {"phase": self.phase, "opened": len(self.opened), "step": "rise", "wait": 0, "tries": 0}
        if self.phase == "open":
            closed = [n for n in DRAWERS if n not in self.opened]
            plan["drawer"] = closed[int(self.rng.integers(len(closed)))] if closed else self.gem_drawer
        return plan

    def autopilot(self):
        robot = self.robot
        plan = self.autopilot_plan
        if plan is None or plan["phase"] != self.phase or plan["opened"] != len(self.opened):
            plan = self.autopilot_plan = self._new_plan()
        command = np.zeros(7)
        eef = robot.eef_pos()
        if self.phase == "open":
            command = self._autopilot_open(plan, eef)
        elif self.phase == "pick":
            command = self._autopilot_pick(plan, eef)
        elif self.phase == "carry":
            above = np.array([*VAULT, 0.90])
            command, reached = self.steer_to(above, tolerance=0.02)
            command[6] = 1
            if reached or plan["step"] == "drop":
                plan["step"] = "drop"
                command[:3] = 0
                command[6] = -1
        command[:3] += self.rng.normal(0, 0.03, 3)
        return np.clip(command, -1, 1), {}

    def _autopilot_open(self, plan, eef):
        drawer = plan["drawer"]
        handle = self.handle_pos(drawer)
        yaw = 0.0                   # fingers spread along y, pinching the bar (which runs along x)
        step = plan["step"]
        if step == "rise":
            command, reached = self.steer_to([eef[0], eef[1], 0.93], yaw=yaw, tolerance=0.03)
            command[6] = -1
            if eef[2] > 0.90:
                plan["step"] = "above"
        elif step == "above":
            command, reached = self.steer_to(handle + [0, 0, 0.08], yaw=yaw, tolerance=0.012)
            command[6] = -1
            if reached and abs(_wrap(yaw - self.robot.euler[2])) < 0.05:
                plan["step"] = "descend"
        elif step == "descend":
            command, reached = self.steer_to(handle + [0, 0, 0.0], yaw=yaw, tolerance=0.008)
            command[6] = -1
            if reached:
                plan["step"], plan["wait"] = "close", 0
        elif step == "close":
            command = np.zeros(7)
            command[6] = 1
            plan["wait"] += 1
            if plan["wait"] > 8:
                plan["step"], plan["wait"] = "pull", 0
        elif step == "pull":
            closed_x = CABINET_FRONT_X - HANDLE_STANDOFF - 0.012 + GRASP_FROM_KNOB
            goal = np.array([closed_x - DRAWER_TRAVEL + 0.005, handle[1], handle[2]])
            command, _ = self.steer_to(goal, gain=0.6, yaw=yaw)
            command[6] = 1
            plan["wait"] += 1
            q = self.drawer_q(drawer)
            if q >= DRAWER_TRAVEL - 0.015 or (q >= OPEN_M + 0.02 and plan["wait"] > 60):
                plan["step"], plan["wait"] = "release", 0
            elif not any(self.finger_contacts(drawer)) and plan["wait"] > 5:
                plan["step"], plan["wait"] = "release", 0      # slipped: let go and retry
            elif plan["wait"] > 120:
                plan["step"], plan["wait"] = "release", 0
        elif step == "release":
            command = np.zeros(7)
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 6:
                plan["step"] = "rise"
                plan["tries"] += 1
        else:
            command = np.zeros(7)
        return command

    def _autopilot_pick(self, plan, eef):
        pos = self.body_pos("gem")
        rot = self.data.xmat[self.body_id("gem")].reshape(3, 3)
        # Close across a cube face: align the gripper yaw with the gem's yaw modulo 90 degrees.
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
        step = plan["step"]
        if step == "rise":
            command, _ = self.steer_to([eef[0], eef[1], 0.93], yaw=yaw, tolerance=0.03)
            command[6] = -1
            if eef[2] > 0.90:
                plan["step"] = "above"
        elif step == "above":
            command, reached = self.steer_to(pos + [0, 0, 0.09], yaw=yaw, tolerance=0.01)
            command[6] = -1
            if reached and abs(_wrap(yaw - self.robot.euler[2])) < 0.05:
                plan["step"] = "descend"
        elif step == "descend":
            command, reached = self.steer_to(pos + [0, 0, 0.002], yaw=yaw, tolerance=0.008)
            command[6] = -1
            if reached:
                plan["step"], plan["wait"] = "close", 0
        elif step == "close":
            command = np.zeros(7)
            command[6] = 1
            plan["wait"] += 1
            if plan["wait"] > 8:
                plan["step"], plan["wait"] = "lift", 0
        elif step == "lift":
            command, _ = self.steer_to([eef[0], eef[1], self.gem_rest_z + 0.14])
            command[:2] = 0
            command[6] = 1
            plan["wait"] += 1
            if not self.grasped("gem") and plan["wait"] > 5:
                plan["step"], plan["wait"] = "release", 0
        elif step == "release":
            command = np.zeros(7)
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 6:
                plan["step"] = "rise"
        else:
            command = np.zeros(7)
        return command
