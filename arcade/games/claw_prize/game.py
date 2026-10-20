"""Claw Prize: grab a toy and lift it (trainable), carry it to the chute for a rarity roll (game fun)."""
import math

import numpy as np

from arcade.core.game import ArcadeGame, Label, StepControl

TOYS = [
    # name, geom xml (body-local), rest half-height
    ("toy_block", '<geom type="box" size="0.022 0.030 0.022" rgba="0.92 0.30 0.25 1"/>', 0.022),
    ("toy_can", '<geom type="cylinder" size="0.022 0.028" rgba="0.20 0.55 0.95 1"/>', 0.028),
    ("toy_bar", '<geom type="box" size="0.018 0.045 0.018" rgba="0.98 0.78 0.20 1"/>', 0.018),
    ("toy_puck", '<geom type="cylinder" size="0.030 0.018" rgba="0.35 0.80 0.45 1"/>', 0.018),
]
CHUTE = np.array([0.30, -0.33])
CHUTE_HALF = 0.075
GRASP = Label("grasp_lift", "Pick up a toy and lift it.", True)
DELIVER = Label("deliver_prize", "Drop the toy into the prize chute.", False, "game_fun")
RARITY = [("Common", 10, 0.6), ("Rare", 30, 0.27), ("Epic", 60, 0.1), ("Legendary", 150, 0.03)]
LIFT_M = 0.08
HOLD_S = 0.3


class ClawPrize(ArcadeGame):
    name = "claw_prize"
    title = "Claw Prize"
    version = "1"
    port = 8802
    tagline = "Grab a toy, drop it in the chute, roll for a rare prize."
    skills = {"grasp_lift": "Grasp one toy with both fingers and lift it 8 cm for 0.3 s."}
    phases = ["grab", "deliver", "done"]
    controls = [("Goal", "Lift a toy, then drop it into the glowing chute")]
    home_position = (0.45, 0.0, 0.98)
    max_round_frames = 1800
    prizes_per_round = 3

    def objects_xml(self):
        bodies = []
        for i, (name, geom, half) in enumerate(TOYS):
            geom = geom.replace("<geom ", '<geom mass="0.06" friction="1.8 0.02 0.001" condim="4" solref="0.004 1" ')
            bodies.append(f'<body name="{name}" pos="0.5 {0.1 * i - 0.15} {0.74 + half}"><freejoint/>{geom}</body>')
        x, y, h = CHUTE[0], CHUTE[1], CHUTE_HALF
        walls = f'''
        <geom name="chute_floor" type="box" pos="{x} {y} 0.742" size="{h} {h} 0.002" rgba="0.15 0.15 0.18 1"/>
        <geom type="box" pos="{x + h} {y} 0.77" size="0.004 {h} 0.03" rgba="0.25 0.25 0.30 1"/>
        <geom type="box" pos="{x - h} {y} 0.77" size="0.004 {h} 0.03" rgba="0.25 0.25 0.30 1"/>
        <geom type="box" pos="{x} {y + h} 0.77" size="{h} 0.004 0.03" rgba="0.25 0.25 0.30 1"/>
        <geom type="box" pos="{x} {y - h} 0.77" size="{h} 0.004 0.03" rgba="0.25 0.25 0.30 1"/>
        '''
        return "\n".join(bodies) + walls

    def decor_xml(self):
        x, y = CHUTE
        return f'''
        <geom type="box" pos="{x} {y} 0.805" size="0.085 0.085 0.004" rgba="1 0.35 0.75 0.35"/>
        <geom type="box" pos="1.3 0 1.1" size="0.04 1.2 0.5" rgba="0.12 0.08 0.25 1"/>
        <geom type="box" pos="1.26 0 1.45" size="0.01 1.0 0.03" rgba="1 0.35 0.75 1"/>
        <geom type="box" pos="1.26 0 0.82" size="0.01 1.0 0.02" rgba="0.3 0.9 1 1"/>
        <geom type="sphere" pos="1.25 -0.7 1.2" size="0.06" rgba="1 0.85 0.2 1"/>
        <geom type="sphere" pos="1.25 0.7 1.2" size="0.06" rgba="0.3 0.9 1 1"/>
        '''

    def object_bodies(self):
        return [name for name, _, _ in TOYS]

    # ----- rules -----
    def reset(self):
        spots = []
        for name, _, half in TOYS:
            for _ in range(100):
                xy = np.array([self.rng.uniform(0.40, 0.66), self.rng.uniform(-0.12, 0.26)])
                if all(np.linalg.norm(xy - s) > 0.09 for s in spots):
                    break
            spots.append(xy)
            yaw = self.rng.uniform(-math.pi / 2, math.pi / 2)
            self.set_free_body(name, (*xy, 0.74 + half + 0.002), (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        self.phase = "grab"
        self.label = GRASP
        self.delivered = 0
        self.collected = set()
        self.hold_s = 0.0
        self.held = None
        self.rest_z = {}
        self.was_closed = False
        self.last_prize = None
        self.autopilot_plan = None
        self.message = "Grab a toy!"

    def _rest_heights(self):
        return {name: 0.74 + half for name, _, half in TOYS}

    def held_toy(self):
        for name, _, _ in TOYS:
            if name not in self.collected and self.grasped(name):
                return name
        return None

    def in_chute(self, name):
        pos = self.body_pos(name)
        # Prizes pile up in the chute, so height only needs to be below the drop height.
        left, right = self.finger_contacts(name)
        return bool(np.all(np.abs(pos[:2] - CHUTE) < CHUTE_HALF) and pos[2] < 0.88 and not (left or right))

    def after_action(self):
        closed = self.robot.grip > 0.5
        rest = self._rest_heights()
        if self.phase == "grab":
            toy = self.held_toy()
            lift = self.body_pos(toy)[2] - rest[toy] if toy else 0.0
            if toy and lift >= LIFT_M:
                self.hold_s += 1 / 20
                if self.hold_s >= HOLD_S:
                    left, right = self.finger_contacts(toy)
                    self.finish_chunk(True, toy=toy, lift_m=round(float(lift), 4), hold_s=round(self.hold_s, 3),
                                      left_contact=left, right_contact=right)
                    self.held = toy
                    self.phase = "deliver"
                    self.label = DELIVER
                    self.message = "Got it! Drop it in the chute."
            else:
                self.hold_s = 0.0
                if self.was_closed and not closed:
                    # Gripper reopened without a stable lift: that attempt failed.
                    self.finish_chunk(False, reason="released_before_stable_lift")
                    self.message = "Missed — try again."
        elif self.phase == "deliver":
            toy = self.held
            if self.in_chute(toy):
                name, points = self._roll_rarity()
                self.score += points
                self.collected.add(toy)
                self.delivered += 1
                self.last_prize = {"toy": toy, "rarity": name, "points": points}
                self.finish_chunk(True, delivered=toy, rarity=name, points=points)
                self.message = f"{name} prize! +{points}"
                if self.delivered >= self.prizes_per_round or len(self.collected) == len(TOYS):
                    self.phase = "done"
                    self.end_round("round_complete", f"Round over — {int(self.score)} points")
                else:
                    self.phase = "grab"
                    self.label = GRASP
            elif not self.grasped(toy) and self.body_pos(toy)[2] < rest[toy] + 0.01:
                self.finish_chunk(False, reason="dropped_outside_chute")
                self.phase = "grab"
                self.label = GRASP
                self.message = "Dropped it! Grab again."
        self.was_closed = closed

    def _roll_rarity(self):
        roll = self.rng.random()
        for name, points, p in RARITY:
            if roll < p:
                return name, points
            roll -= p
        return RARITY[0][:2]

    def hud(self):
        return {"Prizes": f"{self.delivered}/{self.prizes_per_round}",
                "Last": f"{self.last_prize['rarity']} +{self.last_prize['points']}" if self.last_prize else "—",
                "Hold": f"{self.hold_s:.2f}s"}

    # ----- scripted expert, same command interface as a human -----
    def autopilot(self):
        robot = self.robot
        if self.autopilot_plan is None or self.autopilot_plan.get("phase") != self.phase:
            if self.phase == "grab":
                candidates = [n for n, _, _ in TOYS if n not in self.collected]
                eef = robot.eef_pos()
                toy = min(candidates, key=lambda n: np.linalg.norm(self.body_pos(n)[:2] - eef[:2]))
                self.autopilot_plan = {"phase": "grab", "toy": toy, "step": "above", "wait": 0}
            else:
                self.autopilot_plan = {"phase": self.phase, "step": "carry", "wait": 0}
        plan = self.autopilot_plan
        command = np.zeros(7)
        if self.phase == "grab":
            toy = plan["toy"]
            pos = self.body_pos(toy)
            rot = self.data.xmat[self.body_id(toy)].reshape(3, 3)
            # Close across the narrower horizontal side: fingers spread along the gripper's local y.
            sizes = self.model.geom_size[self.model.body_geomadr[self.body_id(toy)]]
            axis = rot[:, 0] if sizes[0] <= sizes[1] else rot[:, 1]
            yaw = math.atan2(axis[1], axis[0]) - math.pi / 2
            yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
            if plan["step"] == "above":
                command, reached = self.steer_to(pos + [0, 0, 0.10], yaw=yaw, tolerance=0.012)
                command[6] = -1
                if reached and abs(((yaw - robot.euler[2]) + math.pi) % (2 * math.pi) - math.pi) < 0.05:
                    plan["step"] = "descend"
            elif plan["step"] == "descend":
                command, reached = self.steer_to(pos + [0, 0, 0.002], yaw=yaw, tolerance=0.012)
                command[6] = -1
                if reached:
                    plan["step"] = "close"
            elif plan["step"] == "close":
                command[6] = 1
                plan["wait"] += 1
                if plan["wait"] > 8:
                    plan["step"] = "lift"
            elif plan["step"] == "lift":
                command, _ = self.steer_to(robot.target + [0, 0, 0.02])
                command[:2] = 0
                command[6] = 1
                if not self.grasped(toy) and robot.eef_pos()[2] > pos[2] + 0.03:
                    plan["step"] = "above"  # slipped: retry
        else:
            above = np.array([*CHUTE, 0.92])
            command, reached = self.steer_to(above, tolerance=0.02)
            command[6] = 1
            if reached:
                command[6] = -1
        # Human-like: no exact zeros of grip, small noise on motion.
        command[:3] += self.rng.normal(0, 0.03, 3)
        return np.clip(command, -1, 1), {}
