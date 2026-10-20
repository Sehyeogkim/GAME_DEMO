"""Domino Run: stand dominoes on a curved course (trainable), then topple the whole chain (game fun).

Trainable skill `place_domino`: take one domino from the supply tray and stand it upright on the
next glowing marker. The chain only falls all the way when every domino is upright, on its mark
and turned along the course, so careful placement is what the topple payoff rewards.
"""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label, StepControl

TABLE_Z = 0.74
HALF = np.array([0.012, 0.030, 0.045])          # thin (local x), wide (local y), tall (local z)
STAND_Z = TABLE_Z + HALF[2]
DOMINO_MASS = 0.06
NUM_MARKERS = 4
TRAY_DOMINOES = 5
SPACING = 0.075                                 # marker centre spacing along the course, m
TRAY_Y = -0.26
TRAY_X = [0.33 + 0.075 * i for i in range(TRAY_DOMINOES)]
COLORS = ["0.95 0.93 0.86", "0.98 0.55 0.30", "0.40 0.75 0.98", "0.62 0.90 0.45", "0.95 0.45 0.70"]

# Placement thresholds for `place_domino` success.
UPRIGHT_MIN = 0.95
POS_TOL_M = 0.015
YAW_TOL_DEG = 12.0
STILL_S = 0.3
EVAL_TIMEOUT_S = 3.0
FALLEN_COS = 0.7
KNOCKED_COS = 0.85

TOPPLE_WATCH_S = 3.0
POINTS_PER_DOMINO = 20
FULL_CHAIN_BONUS = 50

PLACE = Label("place_domino", "Pick a domino from the tray and stand it on the next glowing marker.", True)
RECOVER = Label("recover_dominoes", "Fallen dominoes return to the tray.", False, "setup")
TOPPLE = Label("topple_chain", "Press Topple and watch the chain fall.", False, "game_fun")

MARKER_DIM = (0.25, 0.65, 0.75, 0.55)
MARKER_NEXT = (1.00, 0.20, 0.85, 1.0)
MARKER_FILLED = (0.45, 0.45, 0.50, 0.6)


def yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def wrap_half_pi(angle):
    """Wrap to [-pi/2, pi/2): dominoes and the gripper are symmetric under a half turn."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


class DominoRun(ArcadeGame):
    name = "domino_run"
    title = "Domino Run"
    version = "1"
    port = 8801
    tagline = "Stand the dominoes on the glowing course, then topple the whole chain."
    skills = {"place_domino": "Pick a domino from the tray and stand it on the next marker: released, "
                              "upright cos > 0.95, position error < 1.5 cm, yaw error < 12 deg, still 0.3 s."}
    phases = ["place", "recover", "topple", "done"]
    controls = [("Goal", "Stand a domino on each glowing marker, then press Topple")]
    action_button = "Topple"
    home_position = (0.42, 0.0, 0.98)
    player_camera = {"pos": (0.05, -0.80, 1.40), "target": (0.50, 0.02, 0.76), "fovy": 50}
    max_round_frames = 2400

    def __init__(self):
        super().__init__()
        self.dominoes = [f"domino_{i}" for i in range(TRAY_DOMINOES)]

    # ----- scene -----
    def objects_xml(self):
        parts = []
        stripe = '<geom type="box" pos="0 0 0" size="{x} 0.0305 0.003" rgba="0.1 0.1 0.12 1" contype="0" conaffinity="0" mass="0.0001"/>'
        for i, name in enumerate(self.dominoes + ["starter"]):
            color = "1.0 0.80 0.15" if name == "starter" else COLORS[i]
            parts.append(
                f'<body name="{name}" pos="{0.3 + 0.08 * i} -0.4 {STAND_Z}"><freejoint/>'
                f'<geom name="{name}_geom" type="box" size="{HALF[0]} {HALF[1]} {HALF[2]}" rgba="{color} 1" '
                f'mass="{DOMINO_MASS}" friction="1.8 0.02 0.001" condim="4" solref="0.004 1"/>'
                + stripe.format(x=HALF[0] + 0.0005) + '</body>')
        # Course markers: a footprint outline per marker, visual only (group 0, seen by the policy).
        # Bodies are static mocap-free children of the world; reset() moves them via body_pos.
        for k in range(NUM_MARKERS):
            hx, hy = HALF[0] + 0.004, HALF[1] + 0.004
            w = 0.0025
            rgba = " ".join(map(str, MARKER_DIM))
            edges = [(0, hy, hx + w, w), (0, -hy, hx + w, w), (hx, 0, w, hy + w), (-hx, 0, w, hy + w)]
            geoms = "".join(
                f'<geom name="marker_{k}_{e}" type="box" pos="{ex} {ey} 0.0006" size="{sx} {sy} 0.0006" '
                f'rgba="{rgba}" contype="0" conaffinity="0"/>' for e, (ex, ey, sx, sy) in enumerate(edges))
            # Centre tick across the thin axis shows the domino's facing direction.
            geoms += (f'<geom name="marker_{k}_4" type="box" pos="0 0 0.0006" size="{hx} 0.0015 0.0006" '
                      f'rgba="{rgba}" contype="0" conaffinity="0"/>')
            parts.append(f'<body name="marker_{k}" mocap="true" pos="0.5 0 {TABLE_Z}">{geoms}</body>')
        # Supply tray: a visual mat under the tray dominoes.
        cx = (TRAY_X[0] + TRAY_X[-1]) / 2
        sx = (TRAY_X[-1] - TRAY_X[0]) / 2 + 0.05
        parts.append(f'<geom name="tray_mat" type="box" pos="{cx} {TRAY_Y} {TABLE_Z + 0.0004}" size="{sx} 0.045 0.0004" '
                     f'rgba="0.30 0.22 0.16 1" contype="0" conaffinity="0"/>')
        return "\n".join(parts)

    def decor_xml(self):
        return '''
        <geom type="box" pos="1.32 0 1.10" size="0.04 1.2 0.5" rgba="0.10 0.12 0.22 1"/>
        <geom type="box" pos="1.28 0 1.48" size="0.01 1.0 0.025" rgba="1.0 0.80 0.15 1"/>
        <geom type="box" pos="1.28 0 0.84" size="0.01 1.0 0.02" rgba="0.20 1.0 0.55 1"/>
        <geom type="cylinder" pos="1.27 -0.75 1.15" size="0.07 0.01" euler="0 90 0" rgba="1 0.45 0.3 1"/>
        <geom type="cylinder" pos="1.27 0.75 1.15" size="0.07 0.01" euler="0 90 0" rgba="0.4 0.75 1 1"/>
        '''

    def object_bodies(self):
        return self.dominoes + ["starter"]

    # ----- state helpers -----
    def _dof(self, name):
        return self.model.jnt_dofadr[self.model.body_jntadr[self.body_id(name)]]

    def is_still(self, name):
        d = self._dof(name)
        v = self.data.qvel[d:d + 6]
        return bool(np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.15)

    def domino_yaw(self, name):
        axis = self.data.xmat[self.body_id(name)].reshape(3, 3)[:, 0]
        return math.atan2(axis[1], axis[0])

    def placement_error(self, name, k):
        pos = self.body_pos(name)
        pos_err = float(np.linalg.norm(pos[:2] - self.markers[k][:2]))
        yaw_err = abs(math.degrees(wrap_half_pi(self.domino_yaw(name) - self.markers[k][2])))
        return pos_err, yaw_err

    def held_domino(self):
        for name in self.dominoes:
            if self.grasped(name):
                return name
        return None

    def next_marker(self):
        for k in range(NUM_MARKERS):
            if self.filled[k] is None:
                return k
        return None

    def needs_return(self, name):
        if name in self.filled or self.grasped(name):
            return False
        return self.body_upright(name) < FALLEN_COS or self.body_pos(name)[2] < TABLE_Z - 0.05

    def _paint_markers(self):
        nxt = self.next_marker()
        for k in range(NUM_MARKERS):
            rgba = MARKER_FILLED if self.filled[k] else (MARKER_NEXT if k == nxt else MARKER_DIM)
            for e in range(5):
                self.model.geom_rgba[self.geom_id(f"marker_{k}_{e}")] = rgba

    # ----- rules -----
    def reset(self):
        # Gentle randomized curve heading roughly toward -y, starting near the left edge of the table.
        heading = -math.pi / 2 + self.rng.uniform(-0.25, 0.25)
        turn = self.rng.uniform(-0.14, 0.14)
        start = np.array([self.rng.uniform(0.44, 0.52), self.rng.uniform(0.25, 0.29)])
        self.starter_pose = (start.copy(), heading)
        points = []
        pos, h = start.copy(), heading
        for _ in range(NUM_MARKERS):
            h_next = h + turn
            mid = (h + h_next) / 2
            pos = pos + SPACING * np.array([math.cos(mid), math.sin(mid)])
            h = h_next
            points.append((pos[0], pos[1], h))
        self.markers = [np.array(p) for p in points]
        for k, (x, y, h) in enumerate(points):
            mocap = self.model.body_mocapid[self.body_id(f"marker_{k}")]
            self.data.mocap_pos[mocap] = (x, y, TABLE_Z)
            self.data.mocap_quat[mocap] = yaw_quat(h)
        self.set_free_body("starter", (*start, STAND_Z + 0.001), yaw_quat(heading))
        self.tray_slots = {}
        order = self.rng.permutation(TRAY_DOMINOES)
        for i, name in zip(order, self.dominoes):
            slot = (TRAY_X[i] + self.rng.normal(0, 0.003), TRAY_Y + self.rng.normal(0, 0.003),
                    math.pi / 2 + self.rng.uniform(-0.15, 0.15))
            self.tray_slots[name] = slot
            self.set_free_body(name, (slot[0], slot[1], STAND_Z + 0.001), yaw_quat(slot[2]))
        self.filled = [None] * NUM_MARKERS
        self.phase = "place"
        self.label = PLACE
        self.was_closed = False
        self.held_while_closed = None
        self.evaluating = None
        self.last_error = None
        self.attempts = 0
        self.failures = 0
        self.recover_frames = 0
        self.topple_frames = None
        self.push_frames = 0
        self.chain_result = None
        self.autopilot_plan = None
        self.message = "Stand a domino on the glowing marker."
        self._paint_markers()

    def before_action(self, command, buttons):
        command = np.asarray(command, float)
        if self.phase == "recover" and self.recover_ready:
            # Teleport fallen dominoes back to their tray slots: setup row, marked as assist.
            for name in self.dominoes:
                if self.needs_return(name):
                    x, y, yaw = self.tray_slots[name]
                    self.set_free_body(name, (x, y, STAND_Z + 0.002), yaw_quat(yaw))
            if self.body_upright("starter") < UPRIGHT_MIN:
                start, heading = self.starter_pose
                self.set_free_body("starter", (*start, STAND_Z + 0.002), yaw_quat(heading))
            self.recover_done = True
            return StepControl(command, assist=True)
        if self.phase == "topple" and self.topple_frames is None and buttons.get("action"):
            self.topple_frames = 0
            self.push_frames = 0
            self.message = "Here it goes!"
        return StepControl(command)

    def substep(self):
        if self.phase == "topple" and self.topple_frames is not None and self.push_frames < 3:
            start, heading = self.starter_pose
            direction = np.array([math.cos(heading), math.sin(heading), 0.0])
            force = direction * 0.6
            lever = np.array([0.0, 0.0, 0.035])
            body = self.body_id("starter")
            self.data.xfrc_applied[body, :3] = force
            self.data.xfrc_applied[body, 3:] = np.cross(lever, force)
            return True
        return False

    def after_action(self):
        closed = self.robot.grip > 0.5
        if self.phase == "place":
            self._after_place(closed)
        elif self.phase == "recover":
            self._after_recover()
        elif self.phase == "topple":
            self._after_topple()
        self.was_closed = closed

    def _fail(self, reason, **evidence):
        self.finish_chunk(False, reason=reason, **evidence)
        self.failures += 1
        self.evaluating = None
        self.held_while_closed = None
        # Any domino or the starter lying down: settle, then return it in a setup chunk.
        if any(self.needs_return(n) for n in self.dominoes) or self.body_upright("starter") < UPRIGHT_MIN:
            self.phase = "recover"
            self.label = RECOVER
            self.recover_frames = 0
            self.recover_ready = False
            self.recover_done = False
            self.message = "Oops! Fallen dominoes go back to the tray."

    def _after_place(self, closed):
        # Knock-over of dominoes already on the course (or the starter) ends this attempt.
        knocked = [self.filled[k] for k in range(NUM_MARKERS)
                   if self.filled[k] and self.body_upright(self.filled[k]) < FALLEN_COS]
        if self.body_upright("starter") < FALLEN_COS:
            knocked.append("starter")
        if knocked:
            for k in range(NUM_MARKERS):
                if self.filled[k] in knocked:
                    self.filled[k] = None
            self._paint_markers()
            self._fail("knocked_over_course", knocked=knocked)
            self.message = "Knocked the course over! Rebuild it."
            return

        if closed:
            held = self.held_domino()
            if held:
                self.held_while_closed = held
                self.evaluating = None
        elif self.was_closed:
            # Gripper just reopened.
            self.attempts += 1
            if self.held_while_closed:
                self.evaluating = {"name": self.held_while_closed, "frames": 0, "still_s": 0.0}
                self.held_while_closed = None
            else:
                self._fail("reopened_without_domino")
                self.message = "Nothing in the gripper — try again."
                return

        if self.evaluating:
            self._evaluate_release()
            return

        # A loose domino lying on the table or off it (dropped while carrying) ends the attempt once still.
        for name in self.dominoes:
            if self.needs_return(name) and name != self.held_while_closed and self.is_still(name):
                self._fail("domino_fell", domino=name)
                return

    def _evaluate_release(self):
        ev = self.evaluating
        name = ev["name"]
        ev["frames"] += 1
        if self.grasped(name):
            self.evaluating = None  # picked up again to adjust
            self.held_while_closed = name
            return
        k = self.next_marker()
        left, right = self.finger_contacts(name)
        if self.is_still(name) and not (left or right):
            ev["still_s"] += 1 / 20
        else:
            ev["still_s"] = 0.0
        timeout = ev["frames"] / 20 >= EVAL_TIMEOUT_S
        if ev["still_s"] < STILL_S and not timeout:
            return
        upright = self.body_upright(name)
        pos_err, yaw_err = self.placement_error(name, k)
        self.last_error = (pos_err, yaw_err)
        evidence = dict(domino=name, marker=k, upright_cos=round(upright, 4), pos_error_m=round(pos_err, 4),
                        yaw_error_deg=round(yaw_err, 2), still_s=round(ev["still_s"], 3),
                        left_contact=bool(left), right_contact=bool(right))
        ok = (not timeout and upright > UPRIGHT_MIN and pos_err < POS_TOL_M and yaw_err < YAW_TOL_DEG)
        if ok:
            self.finish_chunk(True, **evidence)
            self.filled[k] = name
            self.evaluating = None
            self._paint_markers()
            placed = sum(f is not None for f in self.filled)
            self.message = f"Nice! {pos_err * 100:.1f} cm, {yaw_err:.0f} deg off."
            if placed == NUM_MARKERS:
                self.phase = "topple"
                self.label = TOPPLE
                self.message = "Course complete! Press Topple."
            return
        reason = ("not_settled" if timeout else "fell_over" if upright <= UPRIGHT_MIN
                  else "off_marker" if pos_err >= POS_TOL_M else "misaligned")
        self._fail(reason, **evidence)
        if self.phase == "place":
            self.message = f"Placed badly ({reason.replace('_', ' ')}) — pick it up again."

    def _after_recover(self):
        self.recover_frames += 1
        if self.recover_done:
            self.phase = "place"
            self.label = PLACE
            self.message = "Back to it: stand a domino on the glowing marker."
            self._paint_markers()
            return
        moving = [n for n in self.dominoes + ["starter"] if not self.is_still(n)]
        if not moving or self.recover_frames >= 30:
            self.recover_ready = True

    def _after_topple(self):
        if self.topple_frames is None:
            return
        self.topple_frames += 1
        self.push_frames += 1
        if self.topple_frames / 20 < TOPPLE_WATCH_S:
            return
        chain = [n for n in self.filled if n]
        fallen = [n for n in chain if self.body_upright(n) < KNOCKED_COS]
        # Count the leading run: the chain only carries on while each domino hits the next.
        run = 0
        for n in chain:
            if self.body_upright(n) < KNOCKED_COS:
                run += 1
            else:
                break
        full = len(fallen) == len(chain)
        points = POINTS_PER_DOMINO * len(fallen) + (FULL_CHAIN_BONUS if full else 0)
        self.score += points
        self.chain_result = {"fallen": len(fallen), "of": len(chain), "run": run, "full": full, "points": points}
        self.finish_chunk(True, fallen=len(fallen), chain=len(chain), run=run, full_chain=full, points=points)
        self.phase = "done"
        msg = "FULL CHAIN!" if full else f"Chain broke after {run}."
        self.end_round("round_complete", f"{msg} {len(fallen)}/{len(chain)} fell, +{points} points")

    def hud(self):
        placed = sum(f is not None for f in self.filled)
        err = (f"{self.last_error[0] * 100:.1f} cm / {self.last_error[1]:.0f}°" if self.last_error else "—")
        if self.chain_result:
            chain = f"{self.chain_result['fallen']}/{self.chain_result['of']}" + (" FULL" if self.chain_result["full"] else "")
        elif self.phase == "topple":
            chain = "Ready" if self.topple_frames is None else "Falling..."
        else:
            chain = "—"
        return {"Placed": f"{placed}/{NUM_MARKERS}", "Last error": err, "Chain": chain, "Score": int(self.score)}

    # ----- scripted expert, same command interface as a human -----
    def autopilot(self):
        robot = self.robot
        buttons = {}
        plan = self.autopilot_plan
        if plan is None or plan.get("phase") != self.phase:
            plan = self.autopilot_plan = {"phase": self.phase, "step": "choose", "wait": 0, "stuck": 0}
        command = np.zeros(7)
        eef = robot.eef_pos()
        if self.phase == "recover":
            command, _ = self.steer_to([eef[0], eef[1], 1.0], tolerance=0.02)
            command[6] = -1
        elif self.phase == "topple":
            above = np.array([0.30, -0.05, 1.08])
            command, reached = self.steer_to(above, tolerance=0.03)
            command[6] = -1
            if reached and self.topple_frames is None:
                buttons["action"] = True
        elif self.phase == "place":
            command = self._autopilot_place(plan, eef)
        command[:3] += self.rng.normal(0, 0.03, 3)
        return np.clip(command, -1, 1), buttons

    def _grip_yaw_for(self, domino_yaw):
        """Gripper yaw whose finger spread (local y) lies along the domino's thin axis."""
        return wrap_half_pi(domino_yaw - math.pi / 2)

    def _autopilot_place(self, plan, eef):
        robot = self.robot
        command = np.zeros(7)
        step = plan["step"]
        plan["stuck"] += 1
        if plan["stuck"] > 260:
            plan.update(step="retreat", stuck=0)
            step = "retreat"
        if step == "choose":
            k = self.next_marker()
            candidates = [n for n in self.dominoes if n not in self.filled and self.body_upright(n) > 0.97
                          and self.body_pos(n)[2] > TABLE_Z]
            if k is None or not candidates:
                command, _ = self.steer_to([eef[0], eef[1], 1.0])
                command[6] = -1
                return command
            name = min(candidates, key=lambda n: np.linalg.norm(self.body_pos(n)[:2] - eef[:2]))
            plan.update(name=name, marker=k, step="above", wait=0, stuck=0)
            step = "above"
        name = plan.get("name")
        pos = self.body_pos(name)
        if step == "above":
            yaw = self._grip_yaw_for(self.domino_yaw(name))
            command, reached = self.steer_to([pos[0], pos[1], 1.0], yaw=yaw, tolerance=0.012)
            command[6] = -1
            if reached and abs(wrap_half_pi(yaw - robot.euler[2])) < 0.04:
                plan["step"] = "descend"
        elif step == "descend":
            yaw = self._grip_yaw_for(self.domino_yaw(name))
            command, reached = self.steer_to(pos + [0, 0, 0.02], yaw=yaw, tolerance=0.008)
            command[6] = -1
            if reached:
                plan.update(step="close", wait=0)
        elif step == "close":
            command[6] = 1
            plan["wait"] += 1
            if plan["wait"] > 8:
                plan.update(step="lift", offset=eef - pos)
        elif step in ("lift", "carry", "lower", "settle"):
            if not self.grasped(name):
                plan.update(step="retreat", stuck=0)
                command[6] = -1
                return command
            k = plan["marker"]
            mx, my, mh = self.markers[k]
            # Closed loop on the held domino: aim its centre and yaw, not the gripper's.
            yaw = robot.euler[2] + wrap_half_pi(mh - self.domino_yaw(name))
            offset = eef - pos
            goal_xy = np.array([mx, my]) + offset[:2]
            if step == "lift":
                command, reached = self.steer_to([eef[0], eef[1], 1.0], tolerance=0.02)
                if eef[2] > 0.95:
                    plan["step"] = "carry"
            elif step == "carry":
                command, reached = self.steer_to([*goal_xy, 1.0], yaw=yaw, tolerance=0.006)
                if reached and abs(wrap_half_pi(mh - self.domino_yaw(name))) < math.radians(3):
                    plan.update(step="lower", wait=0)
            elif step == "lower":
                place_z = STAND_Z + offset[2] + 0.003
                fast = eef[2] > place_z + 0.04
                command, reached = self.steer_to([*goal_xy, place_z], yaw=yaw, gain=1.0 if fast else 0.3,
                                                 tolerance=0.004)
                if reached or pos[2] < STAND_Z + 0.001:
                    plan.update(step="settle", wait=0)
            else:
                command, _ = self.steer_to(robot.target, yaw=yaw)
                plan["wait"] += 1
                if plan["wait"] > 4:
                    plan.update(step="open", wait=0)
            command[6] = 1
        elif step == "open":
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 6:
                plan.update(step="rise", wait=0)
        elif step == "rise":
            command, _ = self.steer_to([eef[0], eef[1], 1.0], gain=0.5, tolerance=0.02)
            command[6] = -1
            if eef[2] > 0.92 and not self.evaluating:
                plan.update(step="choose", stuck=0)
        elif step == "retreat":
            command, reached = self.steer_to([eef[0], eef[1], 1.0], tolerance=0.02)
            command[6] = -1
            if reached and not self.evaluating:
                plan.update(step="choose", stuck=0)
        return command
