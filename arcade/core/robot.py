"""Panda end-effector controller: normalized delta command -> absolute target -> IK joint targets."""
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_HZ = 20
DT = 1 / CONTROL_HZ
# Per control step at command = 1.0: metres for xyz, radians for roll/pitch/yaw.
ACTION_SCALE = np.array([0.012, 0.012, 0.012, 0.035, 0.035, 0.035])
REST_JOINTS = np.array([0, -0.45, 0, -2.15, 0, 1.80, 0.7854])
GRIPPER_OPEN_CTRL = 255.0
STATE_ORDER = ["joint_pos(7)", "joint_vel(7)", "gripper_width(1)", "eef_pos(3)", "eef_quat_wxyz(4)"]
COMMAND_ORDER = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
# Top-down gripper with fingers spread along world y at yaw = 0.
DOWN_EULER = np.array([math.pi, 0.0, 0.0])


class PandaController:
    def __init__(self, model, data, bounds_low, bounds_high, lock_roll_pitch=True):
        self.model = model
        self.data = data
        self.ik = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.bounds_low = np.asarray(bounds_low, float)
        self.bounds_high = np.asarray(bounds_high, float)
        self.lock_roll_pitch = lock_roll_pitch
        self.target = np.zeros(3)
        self.euler = DOWN_EULER.copy()
        self.grip = 0.0  # 0 open .. 1 closed
        self.joint_target = REST_JOINTS.copy()

    def home(self, position, yaw=0.0, settle_steps=200):
        """Place the arm at `position` pointing down, then let physics settle."""
        self.target = np.clip(np.asarray(position, float), self.bounds_low, self.bounds_high)
        self.euler = DOWN_EULER + [0, 0, yaw]
        self.grip = 0.0
        self.data.qpos[:7] = REST_JOINTS
        self.data.qpos[7:9] = 0.04
        mujoco.mj_forward(self.model, self.data)
        self.joint_target = self.solve_ik(iterations=150)
        self.data.qpos[:7] = self.joint_target
        self.data.qvel[:] = 0
        self.write_ctrl()
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def target_quat(self):
        x, y, z, w = Rotation.from_euler("xyz", self.euler).as_quat()
        return np.array([w, x, y, z])

    def apply_command(self, command):
        """Integrate a clipped 7D command into absolute targets and write actuator controls."""
        command = np.clip(np.asarray(command, float), -1, 1).copy()
        if self.lock_roll_pitch:
            command[3:5] = 0.0
        self.target = np.clip(self.target + command[:3] * ACTION_SCALE[:3], self.bounds_low, self.bounds_high)
        self.euler = self.euler + command[3:6] * ACTION_SCALE[3:]
        self.euler[2] = (self.euler[2] + math.pi) % (2 * math.pi) - math.pi
        self.grip = 1.0 if command[6] > 0 else 0.0
        self.joint_target = self.solve_ik()
        self.write_ctrl()
        return command

    def write_ctrl(self):
        self.data.ctrl[:7] = self.joint_target
        self.data.ctrl[7] = GRIPPER_OPEN_CTRL * (1.0 - self.grip)

    def solve_ik(self, iterations=10):
        d = self.ik
        d.qpos[:] = self.data.qpos
        desired = Rotation.from_euler("xyz", self.euler).as_matrix()
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos = d.site_xpos[self.site]
            rotation = d.site_xmat[self.site].reshape(3, 3)
            rotation_error = Rotation.from_matrix(desired @ rotation.T).as_rotvec()
            error = np.r_[np.clip(self.target - pos, -0.05, 0.05), rotation_error * 0.5]
            if np.linalg.norm(error) < 2e-4:
                break
            mujoco.mj_jacSite(self.model, d, jp, jr, self.site)
            jac = np.vstack([jp[:, :7], jr[:, :7] * 0.5])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 0.002, error)
            # Mild pull toward the rest posture in the null space keeps the elbow tidy.
            null = np.eye(7) - np.linalg.pinv(jac) @ jac
            delta += null @ (REST_JOINTS - d.qpos[:7]) * 0.05
            d.qpos[:7] = np.clip(d.qpos[:7] + np.clip(delta, -0.15, 0.15),
                                 self.model.jnt_range[:7, 0] + 0.01, self.model.jnt_range[:7, 1] - 0.01)
        return d.qpos[:7].copy()

    def eef_pos(self):
        return self.data.site_xpos[self.site].copy()

    def eef_quat(self):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.site])
        return q

    def gripper_width(self):
        return float(self.data.qpos[7] + self.data.qpos[8])

    def policy_state(self):
        return {"joint_pos": self.data.qpos[:7].copy(), "joint_vel": self.data.qvel[:7].copy(),
                "gripper_width": np.array([self.gripper_width()]), "eef_pos": self.eef_pos(),
                "eef_quat": self.eef_quat()}
