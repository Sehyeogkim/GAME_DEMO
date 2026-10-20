"""Push Golf: push the puck onto the tee with a closed gripper (trainable), then aim and putt (game fun)."""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label, StepControl

TABLE_Z = 0.74
PUCK_R = 0.03
PUCK_HALF = 0.012
PUCK_MASS = 0.08
TEE_R = 0.03                       # drawn tee circle
TEE_TOLERANCE = 0.025              # success: puck centre within this of the tee centre
REST_SPEED = 0.01                  # m/s
HOLD_S = 0.3
PUSH_TIMEOUT_S = 20.0
HOLE_R = 0.035                     # puck centre inside this radius and slow enough -> holed
CAPTURE_SPEED = 0.8                # m/s, faster pucks skip over the cup
GREEN_LOW = np.array([0.20, -0.36])
GREEN_HIGH = np.array([0.80, 0.36])
MAX_PUTTS = 3
AIM_ERROR_PER_M = 6.0              # rad of hook per metre of tee offset (quality coupling)
PUTT_OVERSHOOT = 0.05              # auto power aims this far past the hole centre
PUTT_DECEL = 3.5                   # m/s^2, measured sliding deceleration of the puck on the table
PUTT_SUBSTEPS = 10
RESET_FRAMES = 8
BUMPERS = 3
PUSH_HEIGHT = 0.756                # grasp_site height that puts closed fingertips at puck mid-height

PUSH = Label("push_to_tee", "Close the gripper and push the puck so it stops inside the tee circle.", True)
PUTT = Label("aim_putt", "Aim with the gripper yaw and press Putt.", False, "game_fun")
RESET = Label("reset", "The puck is placed for a new attempt.", False, "game_fun")
POINTS = {1: 100, 2: 60, 3: 30}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def segment_distance(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


class PushGolf(ArcadeGame):
    name = "push_golf"
    title = "Push Golf"
    version = "1"
    port = 8808
    tagline = "Push the puck onto the tee, line up the putt, sink it in one."
    skills = {"push_to_tee": "Non-prehensile push: the puck rests with its centre within 2.5 cm of the tee centre, "
                             "speed < 1 cm/s, no finger contact, for 0.3 s."}
    phases = ["push", "aim", "rolling", "reset", "done"]
    controls = [("Goal", "Push the puck onto the yellow tee with the closed gripper"),
                ("Aim", "Gripper yaw (Q/E) sets the putt direction"),
                ("Putt", "Enter / right click putts; a centred puck rolls straight")]
    action_button = "Putt"
    bounds_low = (0.15, -0.42, 0.752)
    home_position = (0.42, 0.0, 0.95)
    max_round_frames = 2400
    player_camera = {"pos": (-0.25, -0.70, 1.55), "target": (0.50, 0.0, 0.74), "fovy": 52}

    # ----- scene -----
    def objects_xml(self):
        gx, gy = (GREEN_LOW + GREEN_HIGH) / 2
        hx, hy = (GREEN_HIGH - GREEN_LOW) / 2
        vis = 'contype="0" conaffinity="0"'
        bumpers = "\n".join(
            f'<body name="bumper{i}" mocap="true" pos="0.2 {-0.5 + 0.1 * i} 0.752">'
            f'<geom type="box" size="0.045 0.010 0.012" rgba="0.85 0.45 0.15 1" friction="0.4 0.005 0.0001"/></body>'
            for i in range(BUMPERS))
        return f'''
        <geom name="green" type="box" pos="{gx} {gy} {TABLE_Z + 0.0002}" size="{hx} {hy} 0.0002" rgba="0.25 0.62 0.30 1" {vis}/>
        <geom name="fringe" type="box" pos="{gx} {gy} {TABLE_Z + 0.0001}" size="{hx + 0.012} {hy + 0.012} 0.0001" rgba="0.95 0.95 0.92 1" {vis}/>
        <body name="tee" mocap="true" pos="0.5 0 {TABLE_Z}">
          <geom type="cylinder" size="{TEE_R + 0.004} 0.0004" pos="0 0 0.0006" rgba="1.0 0.85 0.10 1" {vis}/>
          <geom type="cylinder" size="{TEE_R - 0.004} 0.0004" pos="0 0 0.0010" rgba="0.25 0.62 0.30 1" {vis}/>
          <geom type="cylinder" size="0.004 0.0004" pos="0 0 0.0014" rgba="1.0 0.85 0.10 1" {vis}/>
        </body>
        <body name="hole" mocap="true" pos="0.6 0.2 {TABLE_Z}">
          <geom type="cylinder" size="{HOLE_R + 0.006} 0.0004" pos="0 0 0.0006" rgba="0.95 0.95 0.95 1" {vis}/>
          <geom type="cylinder" size="{HOLE_R} 0.0004" pos="0 0 0.0010" rgba="0.03 0.03 0.03 1" {vis}/>
        </body>
        <body name="puck" pos="0.4 0 {TABLE_Z + PUCK_HALF}">
          <freejoint/>
          <geom name="puck" type="cylinder" size="{PUCK_R} {PUCK_HALF}" mass="{PUCK_MASS}" rgba="0.92 0.15 0.18 1"
                friction="0.35 0.005 0.0001" priority="1" condim="3" solref="0.004 1"/>
          <geom type="cylinder" size="{PUCK_R * 0.45} 0.0006" pos="0 0 {PUCK_HALF}" rgba="1 1 1 1" {vis}/>
        </body>
        {bumpers}
        '''

    def decor_xml(self):
        return '''
        <body name="flag" mocap="true" pos="0.6 0.2 0.74">
          <geom type="cylinder" size="0.003 0.09" pos="0 0 0.09" rgba="0.9 0.9 0.9 1"/>
          <geom type="box" size="0.03 0.0015 0.018" pos="0.03 0 0.16" rgba="1 0.2 0.3 1"/>
        </body>
        <body name="aim_arrow" mocap="true" pos="0 0 -1">
          <geom type="capsule" fromto="0.04 0 0.003 0.20 0 0.003" size="0.004" rgba="1 1 1 0.9"/>
          <geom type="sphere" pos="0.20 0 0.003" size="0.011" rgba="1 0.9 0.2 1"/>
        </body>
        <geom type="box" pos="1.3 0 1.0" size="0.04 1.2 0.45" rgba="0.55 0.78 0.95 1"/>
        <geom type="box" pos="1.26 0 0.80" size="0.01 1.2 0.05" rgba="0.18 0.45 0.22 1"/>
        <geom type="sphere" pos="1.25 -0.6 1.3" size="0.07" rgba="1 0.9 0.4 1"/>
        '''

    def object_bodies(self):
        return ["puck", "tee", "hole"] + [f"bumper{i}" for i in range(BUMPERS)]

    # ----- layout -----
    def _mocap(self, body):
        return self.model.body_mocapid[self.body_id(body)]

    def _place_mocap(self, body, xy, z, yaw=0.0):
        mid = self._mocap(body)
        self.data.mocap_pos[mid] = [xy[0], xy[1], z]
        self.data.mocap_quat[mid] = yaw_quat(yaw)

    def _inside(self, xy, low, high):
        return bool(np.all(xy >= low) and np.all(xy <= high))

    def _sample_course(self):
        for _ in range(500):
            tee = np.array([self.rng.uniform(0.40, 0.56), self.rng.uniform(-0.12, 0.12)])
            angle = self.rng.uniform(-math.pi, math.pi)
            line = np.array([math.cos(angle), math.sin(angle)])
            hole = tee + line * self.rng.uniform(0.22, 0.30)
            if not self._inside(hole, [0.28, -0.29], [0.72, 0.29]):
                continue
            normal = np.array([-line[1], line[0]])
            length = np.linalg.norm(hole - tee)
            side = self.rng.choice([-1.0, 1.0])
            bumpers = [  # two guards beside the line, one backstop behind the cup
                (tee + line * length * self.rng.uniform(0.35, 0.5) + normal * side * self.rng.uniform(0.060, 0.075), angle),
                (tee + line * length * self.rng.uniform(0.65, 0.8) - normal * side * self.rng.uniform(0.060, 0.075), angle),
                (hole + line * self.rng.uniform(0.075, 0.09), angle + math.pi / 2),
            ]
            if not all(self._inside(b, GREEN_LOW + 0.05, GREEN_HIGH - 0.05) for b, _ in bumpers):
                continue
            self.tee, self.hole, self.line_angle, self.bumpers = tee, hole, angle, bumpers
            puck = self._sample_puck(avoid=None)
            if puck is not None:
                return puck
        raise RuntimeError("could not lay out the course")

    def _sample_puck(self, avoid=None):
        """A start spot 15-30 cm from the tee, on the far side from the hole, with a clear push lane."""
        line = np.array([math.cos(self.line_angle), math.sin(self.line_angle)])
        for _ in range(300):
            angle = self.rng.uniform(-math.pi, math.pi)
            away = np.array([math.cos(angle), math.sin(angle)])
            if np.dot(away, line) > 0.3:
                continue
            puck = self.tee + away * self.rng.uniform(0.15, 0.30)
            behind = puck + away * 0.07
            if not self._inside(puck, [0.30, -0.26], [0.66, 0.26]):
                continue
            if not self._inside(behind, [0.24, -0.32], [0.72, 0.32]):
                continue
            if np.linalg.norm(puck - self.hole) < 0.12:
                continue
            if any(segment_distance(b, behind, self.tee) < 0.09 for b, _ in self.bumpers):
                continue
            if avoid is not None and np.linalg.norm(puck - avoid) < 0.10:
                continue
            return puck
        return None

    def reset(self):
        puck = self._sample_course()
        self._place_mocap("tee", self.tee, TABLE_Z)
        self._place_mocap("hole", self.hole, TABLE_Z)
        self._place_mocap("flag", self.hole, TABLE_Z)
        self._place_mocap("aim_arrow", [0, 0], -1)
        for i, (xy, yaw) in enumerate(self.bumpers):
            self._place_mocap(f"bumper{i}", xy, TABLE_Z + 0.012, yaw)
        self.set_free_body("puck", (*puck, TABLE_Z + PUCK_HALF + 0.001))
        self.phase = "push"
        self.label = PUSH
        self.strokes = 0
        self.push_attempts = 1
        self.push_frames = 0
        self.hold_s = 0.0
        self.respawn_at = None
        self.reset_frames = 0
        self.putt_force = None
        self.putt_substeps = 0
        self.roll_frames = 0
        self.slow_frames = 0
        self.last_putt = None
        self.best_miss = None
        self.autopilot_plan = None
        self.message = "Push the puck onto the yellow tee."

    # ----- state helpers -----
    def puck_xy(self):
        return self.body_pos("puck")[:2]

    def puck_speed(self):
        body = self.body_id("puck")
        dof = self.model.jnt_dofadr[self.model.body_jntadr[body]]
        return float(np.linalg.norm(self.data.qvel[dof:dof + 3]))

    def tee_offset(self):
        return float(np.linalg.norm(self.puck_xy() - self.tee))

    def on_green(self):
        return self._inside(self.puck_xy(), GREEN_LOW, GREEN_HIGH) and self.body_pos("puck")[2] > TABLE_Z

    def aim_angle(self):
        """Putt direction = the gripper's yaw (hand x axis points along it)."""
        return float(self.robot.euler[2])

    def _start_reset(self, spot=None):
        self.respawn_at = spot if spot is not None else self._sample_puck(avoid=self.robot.eef_pos()[:2])
        if self.respawn_at is None:
            self.respawn_at = self._sample_puck()
        self.phase = "reset"
        self.label = RESET
        self.reset_frames = 0
        self._place_mocap("aim_arrow", [0, 0], -1)

    # ----- rules -----
    def before_action(self, command, buttons):
        if self.phase == "aim" and (buttons.get("action") or buttons.get("putt")):
            offset = self.puck_xy() - self.tee
            aim = self.aim_angle()
            direction = np.array([math.cos(aim), math.sin(aim)])
            # Quality coupling: an off-centre puck hooks toward the side it sits on.
            side = float(np.sign(direction[0] * offset[1] - direction[1] * offset[0]))
            hook = AIM_ERROR_PER_M * float(np.linalg.norm(offset)) * side
            actual = aim + hook
            distance = float(np.linalg.norm(self.hole - self.puck_xy())) + PUTT_OVERSHOOT
            speed = math.sqrt(2 * PUTT_DECEL * distance)
            impulse = PUCK_MASS * speed
            duration = PUTT_SUBSTEPS * self.model.opt.timestep
            self.putt_force = np.array([math.cos(actual), math.sin(actual), 0]) * impulse / duration
            self.putt_substeps = PUTT_SUBSTEPS
            self.strokes += 1
            self.last_putt = {"aim_rad": round(aim, 4), "hook_rad": round(hook, 4),
                              "tee_offset_m": round(float(np.linalg.norm(offset)), 4), "speed": round(speed, 3)}
            self.phase = "rolling"
            self.roll_frames = 0
            self.slow_frames = 0
            self.best_miss = None
            self.message = f"Putt {self.strokes}!"
            self._place_mocap("aim_arrow", [0, 0], -1)
        return StepControl(command=np.asarray(command, float))

    def substep(self):
        if self.phase == "rolling" and self.putt_substeps > 0:
            self.data.xfrc_applied[self.body_id("puck"), :3] = self.putt_force
            self.putt_substeps -= 1
            return True
        if self.phase == "reset" and self.respawn_at is not None:
            self.set_free_body("puck", (*self.respawn_at, TABLE_Z + PUCK_HALF + 0.001))
            self.respawn_at = None
            return True
        return False

    def after_action(self):
        if self.phase == "push":
            self._after_push()
        elif self.phase == "aim":
            self._place_mocap("aim_arrow", self.puck_xy(), TABLE_Z, self.aim_angle())
            if self.tee_offset() > 0.04:
                self.finish_chunk(False, reason="puck_left_tee_before_putt")
                self.phase = "push"
                self.label = PUSH
                self.push_frames = 0
                self.hold_s = 0.0
                self.push_attempts += 1
                self.message = "The puck slid off the tee. Push it back."
        elif self.phase == "rolling":
            self._after_roll()
        elif self.phase == "reset":
            self.reset_frames += 1
            if self.reset_frames >= RESET_FRAMES:
                self.phase = "push"
                self.label = PUSH
                self.push_frames = 0
                self.hold_s = 0.0
                self.push_attempts += 1
                self.message = "Push the puck onto the yellow tee."

    def _after_push(self):
        self.push_frames += 1
        offset = self.tee_offset()
        speed = self.puck_speed()
        left, right = self.finger_contacts("puck")
        if offset <= TEE_TOLERANCE and speed < REST_SPEED and not (left or right):
            self.hold_s += 1 / 20
            if self.hold_s >= HOLD_S - 1e-9:
                self.finish_chunk(True, tee_offset_m=round(offset, 4), speed_mps=round(speed, 4),
                                  hold_s=round(self.hold_s, 3), finger_contact=False,
                                  gripper_closed=bool(self.robot.grip > 0.5),
                                  push_s=round(self.push_frames / 20, 2))
                self.phase = "aim"
                self.label = PUTT
                quality = "Perfect" if offset < 0.008 else ("Good" if offset < 0.016 else "Off-centre")
                self.message = f"{quality} tee-up ({offset * 100:.1f} cm). Aim with yaw, then Putt."
            return
        self.hold_s = 0.0
        if not self.on_green():
            self.finish_chunk(False, reason="puck_left_green", push_s=round(self.push_frames / 20, 2))
            self.message = "Off the green! New attempt."
            self._start_reset()
        elif self.body_upright("puck") < 0.9:
            self.finish_chunk(False, reason="puck_tipped", push_s=round(self.push_frames / 20, 2))
            self.message = "Puck tipped over! New attempt."
            self._start_reset()
        elif self.push_frames >= PUSH_TIMEOUT_S * 20:
            self.finish_chunk(False, reason="timeout", tee_offset_m=round(offset, 4))
            self.message = "Time's up! New attempt."
            self._start_reset()

    def _after_roll(self):
        self.roll_frames += 1
        to_hole = float(np.linalg.norm(self.puck_xy() - self.hole))
        speed = self.puck_speed()
        self.best_miss = to_hole if self.best_miss is None else min(self.best_miss, to_hole)
        if self.putt_substeps == 0 and to_hole < HOLE_R and speed < CAPTURE_SPEED:
            points = POINTS[self.strokes]
            self.score += points
            self.finish_chunk(True, holed=True, strokes=self.strokes, **self.last_putt)
            self.phase = "done"
            name = "HOLE IN ONE!" if self.strokes == 1 else f"Holed in {self.strokes}!"
            self.end_round("round_complete", f"{name} +{points}")
            return
        self.slow_frames = self.slow_frames + 1 if speed < 0.005 else 0
        stopped = self.putt_substeps == 0 and self.slow_frames >= 4
        if stopped or not self.on_green() or self.roll_frames >= 100:
            self.finish_chunk(False, holed=False, strokes=self.strokes, closest_m=round(self.best_miss, 4),
                              **self.last_putt)
            if self.strokes >= MAX_PUTTS:
                self.score += 5 if self.best_miss < 0.08 else 0
                self.phase = "done"
                self.end_round("round_complete", f"Out of putts. Closest {self.best_miss * 100:.0f} cm.")
            else:
                self.message = f"Missed by {self.best_miss * 100:.0f} cm. Tee it up again."
                self._start_reset()

    def hud(self):
        return {"Strokes": f"{self.strokes}/{MAX_PUTTS}",
                "To hole": f"{np.linalg.norm(self.puck_xy() - self.hole) * 100:.0f} cm",
                "Tee offset": f"{self.tee_offset() * 100:.1f} cm",
                "Hold": f"{self.hold_s:.2f}s"}

    # ----- scripted expert, same command interface as a human -----
    def autopilot(self):
        robot = self.robot
        plan = self.autopilot_plan
        if plan is None or plan["phase"] != self.phase:
            plan = self.autopilot_plan = {"phase": self.phase, "step": "lift", "wait": 0, "frames": 0,
                                          "aim_noise": float(self.rng.normal(0, 0.01))}
        plan["frames"] += 1
        command = np.zeros(7)
        if self.phase == "push":
            command = self._autopilot_push(plan)
        elif self.phase == "aim":
            aim = math.atan2(*(self.hole - self.puck_xy())[::-1]) + plan["aim_noise"]
            line = np.array([math.cos(aim), math.sin(aim)])
            standby = np.array([*(self.puck_xy() - line * 0.10), 0.88])
            command, reached = self.steer_to(standby, yaw=aim, tolerance=0.015)
            yaw_error = abs((aim - robot.euler[2] + math.pi) % (2 * math.pi) - math.pi)
            if reached and yaw_error < 0.01:
                plan["wait"] += 1
            command[6] = 1
            command[:3] += self.rng.normal(0, 0.02, 3)
            if plan["wait"] >= 3:
                return np.clip(command, -1, 1), {"action": True}
            return np.clip(command, -1, 1), {}
        else:
            # Rolling or reset: hold position, gripper closed.
            command[6] = 1
            command[:3] += self.rng.normal(0, 0.02, 3)
        return np.clip(command, -1, 1), {}

    def _autopilot_push(self, plan):
        robot = self.robot
        puck = self.puck_xy()
        to_tee = self.tee - puck
        distance = float(np.linalg.norm(to_tee))
        direction = unit(to_tee)
        yaw = math.atan2(direction[1], direction[0])
        if plan.get("step") == "push":
            yaw = plan["yaw"]                  # hold the pusher heading chosen on approach
            if distance < 0.03:
                direction = plan["direction"]  # near the tee the line is ill-defined: keep the last one
            else:
                plan["direction"] = direction
        contact = PUCK_R + 0.016          # gripper centre to puck centre when the finger face touches
        behind = puck - direction * (contact + 0.025)
        eef = robot.eef_pos()
        step = plan["step"]
        if step == "lift":
            command, reached = self.steer_to([*eef[:2], 0.84], tolerance=0.02)
            command[:2] = 0
            if eef[2] > 0.80 or reached:
                plan["step"] = "above"
        elif step == "above":
            command, reached = self.steer_to([*behind, 0.84], yaw=yaw, tolerance=0.012)
            yaw_error = abs((yaw - robot.euler[2] + math.pi) % (2 * math.pi) - math.pi)
            if reached and yaw_error < 0.06:
                plan["step"] = "descend"
                plan["direction"] = direction
                plan["yaw"] = yaw
        elif step == "descend":
            command, reached = self.steer_to([*behind, PUSH_HEIGHT], yaw=yaw, tolerance=0.008)
            if reached:
                plan["step"] = "push"
                plan["wait"] = 0
        elif step == "push":
            # Keep the pusher on the puck-to-tee line, a little into the puck, slowing near the tee.
            goal = puck - direction * 0.02
            limit = float(np.clip(distance / 0.012 * 0.6, 0.06, 0.35))
            lateral = eef[:2] - puck
            along = float(np.dot(lateral, -direction))
            side = float(abs(direction[0] * lateral[1] - direction[1] * lateral[0]))
            command, _ = self.steer_to([*goal, PUSH_HEIGHT], yaw=yaw, gain=0.8)
            command[:3] *= min(1.0, limit / max(np.linalg.norm(command[:3]), 1e-9))
            heading = np.array([math.cos(plan["yaw"]), math.sin(plan["yaw"])])
            if side > 0.022 or along < 0.0 or along > contact + 0.05 or np.dot(heading, direction) < 0.94:
                plan["step"] = "lift"          # misaligned: re-approach from behind
            elif distance < 0.004:
                plan["step"] = "retreat"
                plan["retreat_from"] = eef[:2].copy()
                plan["wait"] = 0
        elif step == "retreat":
            back = plan["retreat_from"] - direction * 0.035
            command, reached = self.steer_to([*back, PUSH_HEIGHT + 0.01], yaw=yaw, tolerance=0.01)
            command[:3] = np.clip(command[:3], -0.3, 0.3)
            if reached:
                plan["step"] = "settle"
                plan["wait"] = 0
        elif step == "settle":
            command, _ = self.steer_to(robot.target)
            plan["wait"] += 1
            if plan["wait"] > 20 and distance > 0.012:
                plan["step"] = "lift"
        else:
            command = np.zeros(7)
        command[6] = 1.0
        command[:3] += self.rng.normal(0, 0.02, 3)
        return np.clip(command, -1, 1)
