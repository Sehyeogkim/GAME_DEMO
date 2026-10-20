"""Shape Sorter: insert each peg through its matching lid hole (trainable), then an unlock celebration (game fun)."""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label, StepControl

TABLE_Z = 0.74
PEG_HALF_H = 0.0275                     # all pegs are 5.5 cm tall
PEG_COMMON = 'mass="0.06" friction="1.8 0.02 0.001" condim="4" solref="0.004 1"'
# name, geom xml (body-local), yaw symmetry period (None = round), colour
PEGS = [
    ("peg_square", f'<geom type="box" size="0.022 0.022 {PEG_HALF_H}" rgba="0.92 0.30 0.25 1" {PEG_COMMON}/>',
     math.pi / 2, "0.92 0.30 0.25"),
    ("peg_round", f'<geom type="cylinder" size="0.026 {PEG_HALF_H}" rgba="0.20 0.55 0.95 1" {PEG_COMMON}/>',
     None, "0.20 0.55 0.95"),
    ("peg_slot", f'<geom type="box" size="0.011 0.035 {PEG_HALF_H}" rgba="0.98 0.78 0.20 1" {PEG_COMMON}/>',
     math.pi, "0.98 0.78 0.20"),
]
PEG_NAMES = [p[0] for p in PEGS]

# Sorter box, in its own frame (x along the row of holes). Clearance is 6 mm on every width.
BOX_HALF = (0.135, 0.060)
WALL = 0.006
LID_TOP = TABLE_Z + 0.068               # 0.808
LID_HALF_T = 0.003
HOLE_X = {"peg_square": -0.085, "peg_round": 0.0, "peg_slot": 0.085}
SQUARE_HOLE = (0.025, 0.025)            # half sizes: peg 0.044 -> hole 0.050
SLOT_HOLE = (0.014, 0.038)              # peg 0.022 x 0.070 -> hole 0.028 x 0.076
ROUND_R = 0.029                         # peg d 0.052 -> 16-gon inscribed d 0.058
ROUND_OPENING = 0.034                   # square cut in the plate, filled back to a 16-gon by tiles
ROUND_TILES = 16

INSERT = Label("insert_peg", "Pick up a peg and insert it through its matching hole.", True)
UNLOCK = Label("unlock_celebration", "Lock opens: victory wiggle.", False, "game_fun")
BONUS = Label("final_bonus", "All locks open: final bonus.", False, "game_fun")
RESPAWN = Label("respawn_peg", "A tipped peg is stood back up by the game.", False, "setup")
TIPPED_UPRIGHT = 0.7
TIPPED_S = 1.0
RESPAWN_FRAMES = 10
HOLD_S = 0.3
RELEASE_GRACE_S = 1.6
UNLOCK_FRAMES = 30
BONUS_FRAMES = 40
LIGHT_OFF = "0.25 0.25 0.28 1"


def _wrap(angle, period):
    """Wrap into (-period/2, period/2]."""
    return (angle + period / 2) % period - period / 2


class ShapeSorter(ArcadeGame):
    name = "shape_sorter"
    title = "Shape Sorter"
    version = "1"
    port = 8806
    tagline = "Slot every shape through its hole to pop the locks."
    skills = {"insert_peg": "Grasp a peg and insert it through its matching lid hole; the peg must rest "
                            "inside the box (centre below the lid top, inside the footprint) free of the "
                            "fingers for 0.3 s."}
    phases = ["insert", "unlock", "bonus", "respawn", "done"]
    controls = [("Goal", "Drop each shape through the matching hole in the lid"),
                ("Tip", "Line up the yaw (Q/E) before lowering; lid bumps cost points")]
    home_position = (0.42, 0.0, 0.98)
    max_round_frames = 2000
    player_camera = {"pos": (0.62, 0.62, 1.32), "target": (0.48, -0.12, 0.78), "fovy": 52}

    # ----- scene -----
    def _lid_pieces(self):
        """Box geoms (local x, y, half x, half y) covering the lid except the three openings."""
        hx, hy = BOX_HALF
        openings = [(HOLE_X["peg_square"], SQUARE_HOLE), (HOLE_X["peg_round"], (ROUND_OPENING, ROUND_OPENING)),
                    (HOLE_X["peg_slot"], SLOT_HOLE)]
        pieces = []
        edges = [-hx]
        for cx, (ax, ay) in openings:
            edges += [cx - ax, cx + ax]
        edges.append(hx)
        # Full-depth strips between openings.
        for i in range(0, len(edges), 2):
            x0, x1 = edges[i], edges[i + 1]
            pieces.append(((x0 + x1) / 2, 0.0, (x1 - x0) / 2, hy))
        # Strips beside each opening.
        for cx, (ax, ay) in openings:
            half = (hy - ay) / 2
            pieces.append((cx, ay + half, ax, half))
            pieces.append((cx, -ay - half, ax, half))
        return pieces

    def objects_xml(self):
        hx, hy = BOX_HALF
        wall_h = (LID_TOP - 2 * LID_HALF_T - TABLE_Z) / 2
        wall_z = TABLE_Z + wall_h
        lid_z = LID_TOP - LID_HALF_T
        box_rgba = "0.86 0.84 0.78 1"
        lid_rgba = "0.95 0.93 0.88 1"
        walls = f'''
          <geom name="sorter_wall_xp" type="box" pos="{hx - WALL / 2} 0 {wall_z}" size="{WALL / 2} {hy} {wall_h}" rgba="{box_rgba}"/>
          <geom name="sorter_wall_xn" type="box" pos="{-hx + WALL / 2} 0 {wall_z}" size="{WALL / 2} {hy} {wall_h}" rgba="{box_rgba}"/>
          <geom name="sorter_wall_yp" type="box" pos="0 {hy - WALL / 2} {wall_z}" size="{hx} {WALL / 2} {wall_h}" rgba="{box_rgba}"/>
          <geom name="sorter_wall_yn" type="box" pos="0 {-hy + WALL / 2} {wall_z}" size="{hx} {WALL / 2} {wall_h}" rgba="{box_rgba}"/>
          <geom name="sorter_floor" type="box" pos="0 0 {TABLE_Z + 0.001}" size="{hx - WALL} {hy - WALL} 0.001" rgba="0.30 0.28 0.26 1"/>
        '''
        lid = []
        for x, y, ax, ay in self._lid_pieces():
            lid.append(f'<geom type="box" pos="{x:.5f} {y:.5f} {lid_z}" size="{ax:.5f} {ay:.5f} {LID_HALF_T}" rgba="{lid_rgba}"/>')
        # Round hole: tiles whose inner faces are tangent to radius ROUND_R form a 16-gon.
        outer = ROUND_OPENING * math.sqrt(2) + 0.002
        depth = (outer - ROUND_R) / 2
        width = outer * math.tan(math.pi / ROUND_TILES) + 0.002
        cx = HOLE_X["peg_round"]
        for k in range(ROUND_TILES):
            a = 2 * math.pi * k / ROUND_TILES
            r = ROUND_R + depth
            lid.append(f'<geom type="box" pos="{cx + r * math.cos(a):.5f} {r * math.sin(a):.5f} {lid_z}" '
                       f'euler="0 0 {a:.5f}" size="{depth:.5f} {width:.5f} {LID_HALF_T}" rgba="{lid_rgba}"/>')
        # Colour keys on the lid (visual only, group 0, so the policy cameras see which hole is which).
        keys = []
        for name, _, _, rgb in PEGS:
            keys.append(f'<geom type="box" pos="{HOLE_X[name]} {hy - 0.008} {LID_TOP + 0.0006}" size="0.012 0.004 0.0006" '
                        f'rgba="{rgb} 1" contype="0" conaffinity="0"/>')
        bodies = []
        for i, (name, geom, _, _) in enumerate(PEGS):
            bodies.append(f'<body name="{name}" pos="0.45 {0.1 * i} {TABLE_Z + PEG_HALF_H}"><freejoint/>{geom}</body>')
        return (f'<body name="sorter" pos="0.5 -0.19 0">{walls}'
                f'<body name="sorter_lid">{"".join(lid)}{"".join(keys)}</body></body>' + "".join(bodies))

    def decor_xml(self):
        hx, hy = BOX_HALF
        lights = []
        # Lock lights on posts behind each hole; the panel body follows the box pose in reset().
        for name in PEG_NAMES:
            x, y = HOLE_X[name], hy + 0.025
            lights.append(f'<geom name="light_{name}" type="sphere" pos="{x} {y} {LID_TOP + 0.06}" size="0.016" rgba="{LIGHT_OFF}"/>')
            lights.append(f'<geom type="cylinder" pos="{x} {y} {TABLE_Z + 0.03}" size="0.005 0.03" rgba="0.4 0.4 0.45 1"/>')
            lights.append(f'<geom type="box" pos="{x} {y} {LID_TOP + 0.03}" size="0.012 0.004 0.03" rgba="0.4 0.4 0.45 1"/>')
        return f'<body name="lock_panel" pos="0.5 -0.19 0">{"".join(lights)}</body>' + '''
        <geom type="box" pos="1.3 0 1.1" size="0.04 1.2 0.5" rgba="0.10 0.16 0.22 1"/>
        <geom type="box" pos="1.26 0 1.45" size="0.01 1.0 0.03" rgba="0.30 0.95 0.55 1"/>
        <geom type="box" pos="1.26 0 0.82" size="0.01 1.0 0.02" rgba="1 0.80 0.25 1"/>
        '''

    def object_bodies(self):
        return list(PEG_NAMES)

    # ----- geometry helpers -----
    def to_local(self, xy):
        c, s = math.cos(self.box_yaw), math.sin(self.box_yaw)
        d = np.asarray(xy[:2], float) - self.box_center
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def to_world(self, local):
        c, s = math.cos(self.box_yaw), math.sin(self.box_yaw)
        return self.box_center + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])

    def hole_xy(self, peg):
        return self.to_world((HOLE_X[peg], 0.0))

    def peg_yaw(self, peg):
        rot = self.data.xmat[self.body_id(peg)].reshape(3, 3)
        return math.atan2(rot[1, 0], rot[0, 0])

    def in_box(self, peg):
        pos = self.body_pos(peg)
        local = self.to_local(pos)
        return bool(pos[2] < LID_TOP and abs(local[0]) < BOX_HALF[0] - WALL and abs(local[1]) < BOX_HALF[1] - WALL)

    def _set_light(self, peg, rgba):
        self.model.geom_rgba[self.geom_id(f"light_{peg}")] = rgba

    # ----- rules -----
    def reset(self):
        self.box_center = np.array([self.rng.uniform(0.47, 0.53), self.rng.uniform(-0.21, -0.17)])
        self.box_yaw = math.pi / 2 + self.rng.uniform(-0.25, 0.25)
        sorter = self.body_id("sorter")
        self.model.body_pos[sorter] = [*self.box_center, 0.0]
        self.model.body_quat[sorter] = [math.cos(self.box_yaw / 2), 0, 0, math.sin(self.box_yaw / 2)]
        panel = self.body_id("lock_panel")
        self.model.body_pos[panel] = self.model.body_pos[sorter]
        self.model.body_quat[panel] = self.model.body_quat[sorter]
        for name in PEG_NAMES:
            self._set_light(name, [0.25, 0.25, 0.28, 1])
        spots = []
        for name in PEG_NAMES:
            xy = self._free_spot(spots)
            spots.append(xy)
            self._stand_peg(name, xy)
        self.phase = "insert"
        self.label = INSERT
        self.inserted = []
        self.insert_times = []
        self.last_insert = None
        self.chunk_start = 0
        self.hold = {n: 0.0 for n in PEG_NAMES}
        self.entry_hole = {}
        self.lid_contacts = {n: 0 for n in PEG_NAMES}
        self.touching_lid = {n: False for n in PEG_NAMES}
        self.held_this_chunk = set()
        self.was_closed = False
        self.release_timer = None
        self.fun_frames = 0
        self.tipped = {n: 0.0 for n in PEG_NAMES}
        self.respawn_peg = None
        self.respawns = 0
        self.autopilot_plan = None
        self.message = "Sort the shapes!"

    def _free_spot(self, taken):
        for _ in range(200):
            xy = np.array([self.rng.uniform(0.36, 0.62), self.rng.uniform(0.0, 0.28)])
            if all(np.linalg.norm(xy - np.asarray(s)[:2]) > 0.10 for s in taken):
                break
        return xy

    def _stand_peg(self, name, xy):
        yaw = self.rng.uniform(-math.pi, math.pi)
        self.set_free_body(name, (*xy, TABLE_Z + PEG_HALF_H + 0.002), (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))

    def _new_insert_chunk(self):
        self.phase = "insert"
        self.label = INSERT
        self.chunk_start = self.frame
        self.held_this_chunk = set()
        self.lid_contacts = {n: 0 for n in PEG_NAMES}
        self.release_timer = None

    def _points(self, seconds, contacts):
        speed = math.exp(-max(0.0, seconds - 5.0) / 12.0)
        clean = 0.8 ** contacts
        return int(round(100 * speed * clean)), speed, clean

    def before_action(self, command, buttons):
        if self.phase == "respawn":
            if self.fun_frames == 0:
                # Non-physical reset of a tipped peg: marked assist, inside a non-trainable chunk.
                taken = [self.body_pos(n) for n in PEG_NAMES if n != self.respawn_peg] + [self.robot.eef_pos()]
                self._stand_peg(self.respawn_peg, self._free_spot(taken))
                hold = np.zeros(7)
                hold[6] = -1.0
                return StepControl(command=hold, source="game_script", assist=True)
            hold = np.zeros(7)
            hold[2] = 0.6
            hold[6] = -1.0
            return StepControl(command=hold, source="game_script")
        if self.phase in ("unlock", "bonus"):
            k = self.fun_frames
            wiggle = np.zeros(7)
            amplitude = 0.9 if self.phase == "bonus" else 0.6
            wiggle[5] = amplitude * math.sin(2 * math.pi * k / 10)
            wiggle[2] = 0.5 if k < 8 else 0.0
            wiggle[6] = -1.0
            return StepControl(command=wiggle, source="game_script")
        return StepControl(command=np.asarray(command, float))

    def after_action(self):
        closed = self.robot.grip > 0.5
        remaining = [n for n in PEG_NAMES if n not in self.inserted]
        if self.phase == "insert":
            for name in remaining:
                if self.grasped(name):
                    self.held_this_chunk.add(name)
                lid = self.touching(name, "sorter_lid")
                if lid and not self.touching_lid[name]:
                    self.lid_contacts[name] += 1
                self.touching_lid[name] = lid
                pos = self.body_pos(name)
                if name not in self.entry_hole and pos[2] < LID_TOP + 0.005:
                    local = self.to_local(pos)
                    if abs(local[0]) < BOX_HALF[0] and abs(local[1]) < BOX_HALF[1]:
                        self.entry_hole[name] = min(HOLE_X, key=lambda h: abs(HOLE_X[h] - local[0]))
                left, right = self.finger_contacts(name)
                if self.in_box(name) and not (left or right):
                    self.hold[name] += 1 / 20
                else:
                    self.hold[name] = 0.0
                if self.hold[name] >= HOLD_S:
                    self._on_inserted(name)
                    break
                speed = np.linalg.norm(self.data.cvel[self.body_id(name)][3:])
                if (self.body_upright(name) < TIPPED_UPRIGHT and not (left or right) and speed < 0.05
                        and not self.in_box(name)):
                    self.tipped[name] += 1 / 20
                else:
                    self.tipped[name] = 0.0
                if self.tipped[name] >= TIPPED_S:
                    self.finish_chunk(False, reason="peg_tipped", peg=name,
                                      upright=round(self.body_upright(name), 3))
                    self.tipped[name] = 0.0
                    self.respawn_peg = name
                    self.respawns += 1
                    self.phase = "respawn"
                    self.label = RESPAWN
                    self.fun_frames = 0
                    self.message = "Peg tipped over — standing it back up."
                    break
            else:
                if self.was_closed and not closed:
                    if self.held_this_chunk:
                        self.release_timer = 0.0
                    else:
                        self.finish_chunk(False, reason="opened_without_grasp")
                        self._new_insert_chunk()
                        self.message = "Missed — try again."
                elif closed and not self.was_closed and self.release_timer is not None:
                    self.finish_chunk(False, reason="regrasp_after_release",
                                      lid_contacts=sum(self.lid_contacts.values()))
                    self._new_insert_chunk()
                    self.message = "Regrasp — that attempt is over."
                elif self.release_timer is not None:
                    self.release_timer += 1 / 20
                    if self.release_timer >= RELEASE_GRACE_S:
                        self.finish_chunk(False, reason="dropped_outside_box",
                                          lid_contacts=sum(self.lid_contacts.values()))
                        self._new_insert_chunk()
                        self.message = "Missed the hole! Pick it up again."
        elif self.phase == "respawn":
            self.fun_frames += 1
            if self.fun_frames >= RESPAWN_FRAMES:
                self.respawn_peg = None
                self._new_insert_chunk()
        elif self.phase in ("unlock", "bonus"):
            self.fun_frames += 1
            if self.phase == "bonus":
                on = (self.fun_frames // 4) % 2 == 0
                for name in PEG_NAMES:
                    self._set_light(name, [1, 0.85, 0.2, 1] if on else [0.3, 1, 0.5, 1])
            limit = UNLOCK_FRAMES if self.phase == "unlock" else BONUS_FRAMES
            if self.fun_frames >= limit:
                self.finish_chunk(True, celebration=self.phase, frames=self.fun_frames)
                if self.phase == "bonus":
                    self.phase = "done"
                    self.end_round("round_complete", f"All unlocked — {int(self.score)} points")
                elif len(self.inserted) == len(PEG_NAMES):
                    self.phase = "bonus"
                    self.label = BONUS
                    self.fun_frames = 0
                    clean = sum(1 for t in self.insert_times if t["lid_contacts"] == 0)
                    bonus = 100 + 50 * clean
                    self.score += bonus
                    self.message = f"Jackpot! All three locks open +{bonus}"
                else:
                    self._new_insert_chunk()
                    self.message = "Next shape!"
        self.was_closed = closed

    def _on_inserted(self, name):
        seconds = (self.frame - self.chunk_start) / 20
        contacts = self.lid_contacts[name]
        points, speed, clean = self._points(seconds, contacts)
        hole = self.entry_hole.get(name, "unknown")
        matched = hole == name
        grasped = name in self.held_this_chunk
        pos = self.body_pos(name)
        success = matched and grasped
        self.finish_chunk(success, peg=name, entry_hole=hole, grasped=grasped, seconds=round(seconds, 2),
                          lid_contacts=contacts, peg_z=round(float(pos[2]), 4),
                          hold_s=round(self.hold[name], 3), upright=round(self.body_upright(name), 3))
        self.inserted.append(name)
        self.score += points
        self.last_insert = {"peg": name, "seconds": seconds, "points": points, "lid_contacts": contacts}
        self.insert_times.append(self.last_insert)
        gold = contacts == 0 and seconds < 8.0
        self._set_light(name, [1, 0.85, 0.2, 1] if gold else [0.3, 1, 0.5, 1])
        self.message = (f"{'Perfect' if gold else 'Unlocked'}! +{points} "
                        f"({seconds:.1f}s, {contacts} lid bumps)")
        self.phase = "unlock"
        self.label = UNLOCK
        self.fun_frames = 0
        self.release_timer = None

    def hud(self):
        last = self.last_insert
        return {"Inserted": f"{len(self.inserted)}/{len(PEG_NAMES)}",
                "Last": f"{last['seconds']:.1f}s +{last['points']}" if last else "—",
                "Lid bumps": str(sum(self.lid_contacts.values())) if self.phase == "insert" else "—",
                "Score": str(int(self.score))}

    # ----- scripted expert, same command interface as a human -----
    def _grasp_yaw(self, peg, period):
        if period is None:
            return 0.0
        rot = self.data.xmat[self.body_id(peg)].reshape(3, 3)
        sizes = self.model.geom_size[self.model.body_geomadr[self.body_id(peg)]]
        axis = rot[:, 0] if sizes[0] <= sizes[1] else rot[:, 1]
        yaw = math.atan2(axis[1], axis[0]) - math.pi / 2
        return _wrap(yaw, period)

    def autopilot(self):
        robot = self.robot
        command = np.zeros(7)
        if self.phase != "insert":
            self.autopilot_plan = None
            return command, {}
        plan = self.autopilot_plan
        if plan is None:
            remaining = [n for n in PEG_NAMES if n not in self.inserted]
            eef = robot.eef_pos()
            peg = min(remaining, key=lambda n: np.linalg.norm(self.body_pos(n)[:2] - eef[:2]))
            plan = self.autopilot_plan = {"peg": peg, "step": "above", "wait": 0, "steps": 0}
        plan["steps"] += 1
        peg = plan["peg"]
        period = dict((p[0], p[2]) for p in PEGS)[peg]
        pos = self.body_pos(peg)
        step = plan["step"]
        if step == "above":
            yaw = self._grasp_yaw(peg, period)
            command, reached = self.steer_to(pos + [0, 0, 0.10], yaw=yaw, tolerance=0.01)
            command[6] = -1
            if reached and abs(_wrap(yaw - robot.euler[2], 2 * math.pi)) < 0.04:
                plan["step"] = "descend"
        elif step == "descend":
            yaw = self._grasp_yaw(peg, period)
            command, reached = self.steer_to(pos + [0, 0, 0.014], yaw=yaw, tolerance=0.008)
            command[6] = -1
            if reached:
                plan["step"], plan["wait"] = "close", 0
        elif step == "close":
            command[6] = 1
            plan["wait"] += 1
            if plan["wait"] > 8:
                plan["step"] = "lift"
                plan["offset_z"] = robot.eef_pos()[2] - pos[2]
        elif step in ("lift", "transit", "align", "insert"):
            command[6] = 1
            eef = robot.eef_pos()
            plan["lost"] = 0 if self.grasped(peg) else plan.get("lost", 0) + 1
            dropped = eef[2] - pos[2] > plan["offset_z"] + 0.025
            if (plan["lost"] >= 4 or dropped) and (step != "insert" or pos[2] > LID_TOP + 0.01):
                self.autopilot_plan = None   # slipped: replan (the game closes the chunk on release)
                command[6] = -1
                return command, {}
            hole = self.hole_xy(peg)
            # Place the peg, not the gripper: correct for where the peg sits in the hand.
            target_xy = hole + (eef[:2] - pos[:2])
            offset_z = eef[2] - pos[2]
            if period is None:
                yaw = robot.euler[2]
            else:
                yaw = robot.euler[2] + _wrap(self.box_yaw - self.peg_yaw(peg), period)
            hover_z = LID_TOP + PEG_HALF_H + 0.012 + offset_z
            if step == "lift":
                command, reached = self.steer_to([*robot.target[:2], 0.93], tolerance=0.02)
                command[:2] = 0
                if robot.eef_pos()[2] > hover_z + 0.03:
                    plan["step"] = "transit"
            elif step == "transit":
                command, reached = self.steer_to([*target_xy, max(hover_z + 0.03, 0.90)], yaw=yaw, tolerance=0.015)
                if reached:
                    plan["step"], plan["wait"] = "align", 0
            elif step == "align":
                command, _ = self.steer_to([*target_xy, hover_z], yaw=yaw, gain=0.7, tolerance=0.004)
                err = np.linalg.norm(pos[:2] - hole)
                yaw_err = abs(yaw - robot.euler[2])
                plan["wait"] = plan["wait"] + 1 if (err < 0.0025 and yaw_err < 0.03
                                                    and abs(eef[2] - hover_z) < 0.006) else 0
                if plan["wait"] >= 3:
                    plan["step"] = "insert"
            elif step == "insert":
                bottom_goal = LID_TOP - 0.02
                goal_z = bottom_goal + PEG_HALF_H + offset_z
                command, _ = self.steer_to([*target_xy, goal_z], yaw=yaw, gain=0.6, tolerance=0.004)
                command[2] = max(command[2], -0.35)
                if pos[2] - PEG_HALF_H < bottom_goal + 0.003:
                    plan["step"], plan["wait"] = "release", 0
            command[6] = 1
        elif step == "release":
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 4:
                plan["step"] = "retreat"
        elif step == "retreat":
            command, _ = self.steer_to([*robot.target[:2], 0.95], tolerance=0.02)
            command[:2] = 0
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 50:      # insertion never registered: try again
                self.autopilot_plan = None
        if plan["steps"] > 600:
            self.autopilot_plan = None
        command[:3] += self.rng.normal(0, 0.03, 3)
        return np.clip(command, -1, 1), {}
