"""Bowling Setup: stand three pins on their spots (trainable), then roll the ball (game fun)."""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label, StepControl

TABLE_Z = 0.74
PIN_RADIUS = 0.018
PIN_HALF = 0.05                  # cylinder half-height; body origin is the cylinder centre
PIN_REST_Z = TABLE_Z + PIN_HALF
PINS = ["pin_1", "pin_2", "pin_3"]
# Pin deck: head pin nearest the ball, two back pins. Inside the ball's sweep so a clean set strikes.
SPOTS = np.array([[0.56, 0.0], [0.495, 0.045], [0.495, -0.045]])
SPOT_NAMES = ["head", "left", "right"]
RACK_X = (0.31, 0.62)
RACK_Y = (-0.33, -0.15)
BALL_RADIUS = 0.040
BALL_START = np.array([1.08, 0.0, TABLE_Z + BALL_RADIUS])
BALL_SPEED = 1.7                 # m/s down the lane (-x)

SET_POS_TOL = 0.015
SET_UPRIGHT = 0.95
STILL_S = 0.3
STILL_SPEED = 0.02               # m/s
SETTLE_TIMEOUT_S = 1.5
FALLEN_UPRIGHT = 0.7
KNOCKED_SET_UPRIGHT = 0.9
KNOCKED_SET_MOVE = 0.025
ROLL_WAIT_S = 3.0
CARRY_Z = 0.99
ROLL_CLEAR_Z = 0.90              # gripper must be lifted clear before the ball may be rolled

SET_PIN = Label("set_pin", "Pick up a pin and stand it upright on a free pin spot.", True)
RERACK = Label("rerack_pin", "A fallen pin is returned to the rack.", False, "setup")
ROLL = Label("roll_ball", "Lift the arm clear and press Roll to bowl.", False, "game_fun")


class BowlingSetup(ArcadeGame):
    name = "bowling_setup"
    title = "Bowling Setup"
    version = "1"
    port = 8805
    tagline = "Set the pins yourself, then bowl them down. Straighter pins, bigger strikes."
    skills = {"set_pin": "Grasp a pin from the rack and stand it upright on a free pin spot: released, "
                         "upright cos > 0.95, centre within 1.5 cm of the spot, still for 0.3 s."}
    phases = ["set", "roll", "rolling", "done"]
    controls = [("Goal", "Stand all 3 pins on the glowing spots, then press Roll"),
                ("Roll", "Enter / right click once every pin is set and the arm is lifted")]
    action_button = "Roll"
    home_position = (0.42, -0.10, 0.98)
    max_round_frames = 2000

    # ----- scene -----
    def objects_xml(self):
        friction = 'friction="1.8 0.02 0.001" condim="4" solref="0.004 1"'
        parts = []
        for i, name in enumerate(PINS):
            x = RACK_X[0] + 0.1 * i
            parts.append(f'''
            <body name="{name}" pos="{x} -0.24 {PIN_REST_Z}"><freejoint/>
              <geom name="{name}_body" type="cylinder" size="{PIN_RADIUS} {PIN_HALF}" mass="0.07" {friction}
                    rgba="0.96 0.96 0.94 1"/>
              <geom name="{name}_head" type="sphere" pos="0 0 {PIN_HALF + 0.012}" size="0.016" mass="0.008"
                    {friction} rgba="0.96 0.96 0.94 1"/>
              <geom type="cylinder" pos="0 0 {PIN_HALF - 0.012}" size="{PIN_RADIUS + 0.0006} 0.004"
                    contype="0" conaffinity="0" mass="0.0001" rgba="0.85 0.12 0.12 1"/>
            </body>''')
        bx, by, bz = BALL_START
        parts.append(f'''
        <body name="ball" pos="{bx} {by} {bz}"><freejoint/>
          <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}" mass="0.6" priority="1" friction="0.25 0.005 0.0001"
                rgba="0.15 0.22 0.75 1"/>
        </body>''')
        # Visual-only lane, pin spots and rack outline: the policy cameras must see the targets.
        visual = 'contype="0" conaffinity="0" group="0"'
        parts.append(f'<geom name="lane" type="box" pos="0.80 0 {TABLE_Z + 0.0004}" size="0.36 0.085 0.0004" '
                     f'{visual} rgba="0.86 0.68 0.42 1"/>')
        for i, (x, y) in enumerate(SPOTS):
            parts.append(f'<geom name="spot_{SPOT_NAMES[i]}" type="cylinder" pos="{x} {y} {TABLE_Z + 0.0010}" '
                         f'size="0.024 0.0006" {visual} rgba="0.95 0.20 0.20 1"/>')
        rx, ry = (RACK_X[0] + RACK_X[1]) / 2, (RACK_Y[0] + RACK_Y[1]) / 2
        hx, hy = (RACK_X[1] - RACK_X[0]) / 2 + 0.04, (RACK_Y[1] - RACK_Y[0]) / 2 + 0.04
        parts.append(f'<geom name="rack_mat" type="box" pos="{rx} {ry} {TABLE_Z + 0.0004}" size="{hx} {hy} 0.0004" '
                     f'{visual} rgba="0.35 0.38 0.45 1"/>')
        # Physical gutter rails along the lane and a low backstop behind the pin deck.
        parts.append(f'''
        <geom name="rail_left" type="box" pos="0.86 0.10 {TABLE_Z + 0.012}" size="0.30 0.006 0.012" rgba="0.30 0.30 0.34 1"/>
        <geom name="rail_right" type="box" pos="0.86 -0.10 {TABLE_Z + 0.012}" size="0.30 0.006 0.012" rgba="0.30 0.30 0.34 1"/>
        <geom name="backstop" type="box" pos="0.40 0 {TABLE_Z + 0.015}" size="0.008 0.11 0.015" rgba="0.30 0.30 0.34 1"/>
        ''')
        return "\n".join(parts)

    def decor_xml(self):
        arrows = "".join(
            f'<geom type="box" pos="{0.78 + 0.0 * k} {y} 0.7415" size="0.012 0.004 0.0004" rgba="0.55 0.30 0.12 1"/>'
            for k, y in enumerate((-0.05, -0.025, 0.0, 0.025, 0.05)))
        return f'''
        {arrows}
        <geom type="box" pos="1.00 0 0.7415" size="0.003 0.085 0.0004" rgba="0.2 0.2 0.2 1"/>
        <geom type="cylinder" pos="0.538 0 0.7418" size="0.10 0.0003" rgba="1 0.85 0.25 0.35"/>
        <geom type="box" pos="1.32 0 1.10" size="0.04 1.1 0.45" rgba="0.08 0.06 0.20 1"/>
        <geom type="box" pos="1.28 0 1.42" size="0.01 0.9 0.03" rgba="1 0.30 0.30 1"/>
        <geom type="box" pos="1.28 0 0.84" size="0.01 0.9 0.02" rgba="0.3 0.9 1 1"/>
        <geom type="cylinder" pos="1.27 -0.35 1.12" size="0.05 0.012" euler="0 90 0" rgba="0.96 0.96 0.94 1"/>
        <geom type="cylinder" pos="1.27 0.0 1.12" size="0.05 0.012" euler="0 90 0" rgba="0.96 0.96 0.94 1"/>
        <geom type="cylinder" pos="1.27 0.35 1.12" size="0.05 0.012" euler="0 90 0" rgba="0.96 0.96 0.94 1"/>
        '''

    def object_bodies(self):
        return PINS + ["ball"]

    # ----- state helpers -----
    def pin_speed(self, name):
        body = self.body_id(name)
        joint = self.model.body_jntadr[body]
        dadr = self.model.jnt_dofadr[joint]
        return float(np.linalg.norm(self.data.qvel[dadr:dadr + 3]))

    def free_spots(self):
        return [i for i in range(len(SPOTS)) if i not in self.spot_of.values()]

    def nearest_free_spot(self, xy):
        free = self.free_spots()
        if not free:
            return None, float("inf")
        best = min(free, key=lambda i: np.linalg.norm(SPOTS[i] - xy))
        return best, float(np.linalg.norm(SPOTS[best] - xy))

    def held_pin(self):
        for name in PINS:
            if name not in self.spot_of and self.grasped(name):
                return name
        return None

    def rack_slot(self, avoid=()):
        taken = [self.body_pos(n)[:2] for n in PINS if n not in avoid]
        for _ in range(200):
            xy = np.array([self.rng.uniform(*RACK_X), self.rng.uniform(*RACK_Y)])
            if all(np.linalg.norm(xy - t) > 0.09 for t in taken):
                return xy
        return xy

    # ----- rules -----
    def reset(self):
        placed = []
        for name in PINS:
            for _ in range(200):
                xy = np.array([self.rng.uniform(*RACK_X), self.rng.uniform(*RACK_Y)])
                if all(np.linalg.norm(xy - p) > 0.09 for p in placed):
                    break
            placed.append(xy)
            yaw = self.rng.uniform(-math.pi, math.pi)
            self.set_free_body(name, (*xy, PIN_REST_Z + 0.001), (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        self.set_free_body("ball", BALL_START)
        self.phase = "set"
        self.label = SET_PIN
        self.spot_of = {}             # pin -> spot index once set
        self.placement_error = {}     # pin -> xy error at set time
        self.carrying = None          # pin grasped during this attempt
        self.settling = None          # pin released, waiting for the settle verdict
        self.settle_s = 0.0
        self.still_s = 0.0
        self.was_closed = False
        self.rerack_pending = []
        self.roll_requested = False
        self.ball_launched = False
        self.roll_s = 0.0
        self.roll_aim = float(self.rng.uniform(-0.012, 0.012))  # slight hook per frame
        self.knocked = 0
        self.attempts = 0
        self.autopilot_plan = None
        self.message = "Stand the pins on the red spots!"

    def before_action(self, command, buttons):
        command = np.asarray(command, float)
        if self.rerack_pending:
            # Non-trainable setup row: fallen pins go back to the rack (teleport, so marked assist).
            for name in self.rerack_pending:
                xy = self.rack_slot(avoid=(name,))
                self.set_free_body(name, (*xy, PIN_REST_Z + 0.001))
            self.rerack_pending = []
            return StepControl(command=command, assist=True)
        if self.phase == "roll" and buttons.get("action"):
            if self.robot.eef_pos()[2] >= ROLL_CLEAR_Z:
                self.roll_requested = True
            else:
                self.message = "Lift the arm clear of the lane first!"
        return StepControl(command=command)

    def substep(self):
        if self.roll_requested and not self.ball_launched:
            body = self.body_id("ball")
            dadr = self.model.jnt_dofadr[self.model.body_jntadr[body]]
            direction = np.array([-1.0, self.roll_aim, 0.0])
            direction /= np.linalg.norm(direction)
            velocity = direction * BALL_SPEED
            self.data.qvel[dadr:dadr + 3] = velocity
            # Rolling without slipping: omega = z x v / r.
            self.data.qvel[dadr + 3:dadr + 6] = np.cross([0, 0, 1], velocity) / BALL_RADIUS
            self.ball_launched = True
            return True
        return False

    def after_action(self):
        closed = self.robot.grip > 0.5
        if self.label is RERACK:
            self.label = SET_PIN
        if self.phase == "set":
            self._update_set(closed)
        elif self.phase == "roll":
            if self.ball_launched:
                self.phase = "rolling"
                self.message = "Rolling..."
        elif self.phase == "rolling":
            self.roll_s += 1 / 20
            if self.roll_s >= ROLL_WAIT_S:
                self._score_frame()
        self.was_closed = closed

    def _update_set(self, closed):
        # Already-set pins that get knocked lose their spot; the attempt that knocked them fails.
        for name, spot in list(self.spot_of.items()):
            error = np.linalg.norm(self.body_pos(name)[:2] - SPOTS[spot])
            if self.body_upright(name) < KNOCKED_SET_UPRIGHT or error > KNOCKED_SET_MOVE:
                del self.spot_of[name]
                self.placement_error.pop(name, None)
                self._fail("knocked_set_pin", knocked=name, spot=SPOT_NAMES[spot])
                self.message = "Oops, you knocked a set pin!"
        held = self.held_pin()
        if held is not None:
            if self.settling is not None:
                self.settling = None
            self.carrying = held
        if self.settling is None and self.carrying is not None and self.was_closed and not closed:
            self.settling, self.carrying = self.carrying, None
            self.settle_s = self.still_s = 0.0
        if self.settling is not None:
            self._judge_settle(self.settling)
        elif self.carrying is not None and held is None and not closed:
            self.carrying = None
        # Any pin lying down, off the table, or fallen away from the gripper goes back to the rack.
        for name in PINS:
            if name in self.spot_of or name == self.settling or self.grasped(name):
                continue
            pos = self.body_pos(name)
            if (self.body_upright(name) < FALLEN_UPRIGHT and self.pin_speed(name) < 0.05) or pos[2] < 0.6:
                if name == self.carrying:
                    self.carrying = None
                    self._fail("dropped_pin", pin=name)
                if name not in self.rerack_pending:
                    self.rerack_pending.append(name)
                    self.label = RERACK
        if len(self.spot_of) == len(SPOTS):
            self.phase = "roll"
            self.label = ROLL
            self.carrying = self.settling = None
            self.message = "All pins set! Lift the arm and press Roll."

    def _judge_settle(self, name):
        self.settle_s += 1 / 20
        pos = self.body_pos(name)
        upright = self.body_upright(name)
        spot, error = self.nearest_free_spot(pos[:2])
        left, right = self.finger_contacts(name)
        released = not (left or right) and self.robot.grip < 0.5
        speed = self.pin_speed(name)
        good = released and upright > SET_UPRIGHT and error < SET_POS_TOL and pos[2] < PIN_REST_Z + 0.004
        self.still_s = self.still_s + 1 / 20 if good and speed < STILL_SPEED else 0.0
        if self.still_s >= STILL_S:
            self.spot_of[name] = spot
            self.placement_error[name] = error
            self.attempts += 1
            self.finish_chunk(True, pin=name, spot=SPOT_NAMES[spot], position_error_m=round(error, 4),
                              upright_cos=round(upright, 4), still_s=round(self.still_s, 3),
                              speed_mps=round(speed, 4), released=released)
            self.message = f"Pin set on the {SPOT_NAMES[spot]} spot ({error * 100:.1f} cm)."
            self.settling = None
        elif upright < FALLEN_UPRIGHT or self.settle_s >= SETTLE_TIMEOUT_S:
            reason = "pin_fell" if upright < FALLEN_UPRIGHT else (
                "off_spot" if error >= SET_POS_TOL else "not_upright_or_still")
            self._fail(reason, pin=name, position_error_m=round(min(error, 9.0), 4), upright_cos=round(upright, 4))
            self.message = "Not quite. Try that pin again."
            self.settling = None

    def _fail(self, reason, **evidence):
        self.attempts += 1
        if self.label is SET_PIN and self.pending_result is None:
            self.finish_chunk(False, reason=reason, **evidence)

    def _score_frame(self):
        down = [n for n in PINS if self.body_upright(n) < FALLEN_UPRIGHT]
        self.knocked = len(down)
        strike = self.knocked == len(PINS)
        points = 10 * self.knocked + (20 if strike else 0)
        self.score += points
        mean_error = float(np.mean(list(self.placement_error.values()))) if self.placement_error else 0.0
        self.finish_chunk(True, knocked=self.knocked, strike=strike, points=points,
                          mean_placement_error_m=round(mean_error, 4), roll_aim=round(self.roll_aim, 4))
        self.phase = "done"
        self.message = ("STRIKE! " if strike else "") + f"{self.knocked}/3 pins down, +{points}"
        self.end_round("round_complete", self.message)

    def hud(self):
        return {"Pins set": f"{len(self.spot_of)}/3",
                "Knocked": f"{self.knocked}/3" if self.phase == "done" else "—",
                "Score": f"{int(self.score)}"}

    # ----- scripted expert, same command interface as a human -----
    def autopilot(self):
        robot = self.robot
        command = np.zeros(7)
        buttons = {}
        if self.phase in ("roll", "rolling", "done"):
            above = np.array([0.38, -0.10, 1.02])
            command, reached = self.steer_to(above, yaw=0.0, tolerance=0.03)
            command[6] = -1
            if self.phase == "roll" and robot.eef_pos()[2] >= ROLL_CLEAR_Z + 0.02:
                buttons["action"] = True
            command[:3] += self.rng.normal(0, 0.03, 3)
            return np.clip(command, -1, 1), buttons
        plan = self.autopilot_plan
        if plan is None or plan["pin"] in self.spot_of or plan["pin"] in self.rerack_pending:
            candidates = [n for n in PINS if n not in self.spot_of and n not in self.rerack_pending]
            if not candidates:
                return np.zeros(7) + np.r_[np.zeros(6), -1], buttons
            eef = robot.eef_pos()
            pin = min(candidates, key=lambda n: np.linalg.norm(self.body_pos(n)[:2] - eef[:2]))
            plan = self.autopilot_plan = {"pin": pin, "step": "above", "wait": 0}
        pin = plan["pin"]
        pos = self.body_pos(pin)
        grasp_point = pos + [0, 0, 0.022]
        step = plan["step"]
        if step == "above":
            command, reached = self.steer_to(grasp_point + [0, 0, 0.12], yaw=0.0, tolerance=0.01)
            command[6] = -1
            if reached and self.body_upright(pin) > 0.9:
                plan["step"] = "descend"
        elif step == "descend":
            command, reached = self.steer_to(grasp_point, gain=0.7, yaw=0.0, tolerance=0.006)
            command[6] = -1
            if reached:
                plan["step"], plan["wait"] = "close", 0
        elif step == "close":
            command[6] = 1
            plan["wait"] += 1
            if plan["wait"] > 8 and self.grasped(pin):
                plan["step"] = "lift"
            elif plan["wait"] > 25:
                plan["step"] = "above"
                plan["offset"] = robot.eef_pos() - pos
        elif step in ("lift", "carry", "lower"):
            # Lost the pin: it no longer follows the fingers (single-row contact flicker is ignored).
            lost = (robot.eef_pos()[2] - pos[2] - plan["offset"][2] > 0.02
                    or np.linalg.norm(robot.eef_pos()[:2] - pos[:2] - plan["offset"][:2]) > 0.02
                    or robot.gripper_width() < 0.01)
            if lost:
                plan["step"] = "release_wait" if step == "lower" else "above"
                plan["wait"] = 0
                command[6] = -1
            else:
                spot = plan.get("spot")
                if spot is None or spot not in self.free_spots():
                    free = self.free_spots()
                    spot = plan["spot"] = min(free, key=lambda i: (SPOTS[i][0] < 0.53, -SPOTS[i][0]))
                # Closed loop on where the pin actually sits in the fingers.
                offset = robot.eef_pos() - pos
                target = np.array([*SPOTS[spot], PIN_REST_Z + 0.004]) + offset
                if step == "lift":
                    command, reached = self.steer_to([*robot.target[:2], CARRY_Z], yaw=0.0, tolerance=0.02)
                    command[:2] = 0
                    if reached:
                        plan["step"] = "carry"
                elif step == "carry":
                    command, reached = self.steer_to([*target[:2], CARRY_Z], yaw=0.0, tolerance=0.006)
                    if reached and np.linalg.norm(pos[:2] - SPOTS[spot]) < 0.006:
                        plan["step"] = "lower"
                else:
                    command, reached = self.steer_to(target, gain=0.5, yaw=0.0, tolerance=0.004)
                    xy_error = np.linalg.norm(pos[:2] - SPOTS[spot])
                    if pos[2] < PIN_REST_Z + 0.006 and xy_error < 0.006 and self.pin_speed(pin) < 0.03:
                        plan["step"], plan["wait"] = "release", 0
                command[6] = 1
        elif step == "release":
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 6:
                plan["step"], plan["wait"] = "retreat", 0
        elif step == "retreat":
            command, _ = self.steer_to([*robot.target[:2], 0.95], gain=0.6, yaw=0.0)
            command[:2] = 0
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 40 and self.settling is None:
                plan["step"] = "above"
        elif step == "release_wait":
            command[6] = -1
            plan["wait"] += 1
            if plan["wait"] > 20:
                plan["step"] = "retreat"
        command[:3] += self.rng.normal(0, 0.02, 3)
        return np.clip(command, -1, 1), buttons
