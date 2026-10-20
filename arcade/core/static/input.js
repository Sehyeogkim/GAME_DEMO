// Operator input layer: turns devices into the 7D command [dx, dy, dz, droll, dpitch, dyaw, grip].
// Swap or extend this file for a gamepad; the server only sees the command and buttons.
const KEY_AXES = {
  KeyW: [0, +1], KeyS: [0, -1],      // +x away from the robot base
  KeyA: [1, +1], KeyD: [1, -1],      // +y to the robot's left
  KeyR: [2, +1], KeyF: [2, -1],      // up / down
  KeyQ: [5, +1], KeyE: [5, -1],      // yaw
};
const RAMP_PER_SECOND = 6;           // keys ease in/out so recorded actions are not pure steps
const PRECISION_SCALE = 0.35;
const WHEEL_IMPULSE = 0.9;

export class OperatorInput {
  constructor(target) {
    this.target = target;
    this.held = new Set();
    this.command = [0, 0, 0, 0, 0, 0, -1];
    this.gripClosed = false;
    this.pendingButtons = {};
    this.wheel = 0;
    this.enabled = false;
    this.device = "keyboard_mouse";
    window.addEventListener("keydown", (e) => this.onKey(e, true));
    window.addEventListener("keyup", (e) => this.onKey(e, false));
    window.addEventListener("blur", () => this.releaseAll());
    target.addEventListener("wheel", (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      this.wheel += Math.sign(-e.deltaY) * WHEEL_IMPULSE;
    }, { passive: false });
    target.addEventListener("mousedown", (e) => {
      if (!this.enabled) return;
      if (e.button === 0) this.toggleGrip();
      if (e.button === 2) this.pendingButtons.action = true;
    });
    target.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  onKey(event, down) {
    if (!this.enabled || event.target.closest?.("input, textarea, button")) return;
    if (KEY_AXES[event.code] || ["Space", "Enter", "ShiftLeft", "ShiftRight"].includes(event.code)) event.preventDefault();
    if (down && event.repeat) return;
    if (event.code === "Space" && down) this.toggleGrip();
    else if (event.code === "Enter" && down) this.pendingButtons.action = true;
    else if (down) this.held.add(event.code);
    else this.held.delete(event.code);
  }

  toggleGrip() { this.gripClosed = !this.gripClosed; }

  syncGrip(closed) { this.gripClosed = closed; }

  releaseAll() {
    this.held.clear();
    this.wheel = 0;
    this.command = [0, 0, 0, 0, 0, 0, this.gripClosed ? 1 : -1];
  }

  // Called once per control tick; returns {command, buttons}.
  sample(dt) {
    const goal = [0, 0, 0, 0, 0, 0];
    for (const code of this.held) {
      const axis = KEY_AXES[code];
      if (axis) goal[axis[0]] += axis[1];
    }
    const precise = this.held.has("ShiftLeft") || this.held.has("ShiftRight");
    for (let i = 0; i < 6; i++) {
      let g = Math.max(-1, Math.min(1, goal[i])) * (precise ? PRECISION_SCALE : 1);
      const step = RAMP_PER_SECOND * dt;
      this.command[i] += Math.max(-step, Math.min(step, g - this.command[i]));
      if (Math.abs(this.command[i]) < 1e-3) this.command[i] = 0;
    }
    if (this.wheel !== 0) {
      this.command[2] = Math.max(-1, Math.min(1, this.command[2] + this.wheel));
      this.wheel *= 0.5;
      if (Math.abs(this.wheel) < 0.05) this.wheel = 0;
    }
    this.command[6] = this.gripClosed ? 1 : -1;
    const buttons = this.pendingButtons;
    this.pendingButtons = {};
    return { command: this.command.map((v) => Math.round(v * 1000) / 1000), buttons };
  }
}
