import { OperatorInput } from "/static/input.js";

const $ = (id) => document.getElementById(id);
const view = $("view");
const input = new OperatorInput(view);
const TICK_MS = 50;
let socket = null;
let status = null;
let info = null;

function send(payload) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function connect() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onopen = () => {
    $("conn").textContent = "Connected";
    $("conn").className = "pill ok";
    send({ type: "hello", input_device: input.device });
  };
  socket.onclose = () => {
    $("conn").textContent = "Disconnected — retrying";
    $("conn").className = "pill bad";
    input.enabled = false;
    setTimeout(connect, 1500);
  };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "info") showInfo(message.info);
    else if (message.type === "frame") showFrame(message);
    else if (message.type === "round_over") showRoundOver(message);
    else if (message.type === "error") flash(message.message, "bad");
  };
}

function showInfo(data) {
  info = data;
  document.title = `${data.title} — WEMINE Arcade`;
  $("title").textContent = data.title;
  $("tagline").textContent = data.tagline;
  if (data.action_button) {
    $("action-help").hidden = false;
    $("action-name").textContent = data.action_button;
  }
  for (const [key, text] of data.controls || []) {
    const li = document.createElement("li");
    li.innerHTML = `<b></b> `;
    li.querySelector("b").textContent = key;
    li.append(text);
    $("controls").append(li);
  }
}

function showFrame(message) {
  $("frame").src = `data:image/jpeg;base64,${message.image}`;
  status = message.status;
  $("score").textContent = Math.round(status.score);
  $("time").textContent = `${status.elapsed_s.toFixed(1)} s`;
  const hud = $("hud");
  hud.replaceChildren(...Object.entries(status.hud || {}).map(([k, v]) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = "<span></span><strong></strong>";
    row.children[0].textContent = k;
    row.children[1].textContent = v;
    return row;
  }));
  const label = status.label;
  const chunk = $("chunk");
  chunk.className = `chunk ${label.trainable ? "training" : "other"}`;
  chunk.querySelector(".badge").textContent = label.trainable ? `TRAINING · ${label.skill}` : `OTHER · ${label.other_reason}`;
  chunk.querySelector("p").textContent = label.instruction || "";
  if (status.recording) {
    $("frames").textContent = status.recording.frames;
    $("chunks").textContent = `${status.recording.training_chunks} / ${status.recording.other_chunks}`;
  }
  $("grip").textContent = `Gripper: ${status.grip_closed ? "closed" : "open"}`;
  $("pause").textContent = status.paused ? "Resume" : "Pause";
  if (status.message) $("banner").textContent = status.message;
  input.enabled = status.running && !status.paused;
  $("overlay").hidden = status.running;
}

function showRoundOver(message) {
  input.enabled = false;
  input.releaseAll();
  $("overlay").hidden = false;
  $("overlay-title").textContent = `Round over — ${Math.round(message.score)} points`;
  $("overlay-text").textContent = message.message || "";
  $("start").textContent = "Play again";
  if (message.saved) {
    const s = message.saved;
    $("saved-card").hidden = false;
    $("saved").textContent = `${s.episode_id} · ${s.frames} frames · ${s.training_chunks} training / ${s.other_chunks} other chunks · ${s.end_reason}`;
  }
}

function flash(text, kind) {
  $("banner").textContent = text;
  $("banner").className = `banner ${kind || ""}`;
}

$("start").onclick = () => {
  input.syncGrip(false);
  input.releaseAll();
  send({ type: "start" });
  view.focus();
};
$("pause").onclick = () => {
  if (!status) return;
  input.releaseAll();
  send({ type: status.paused ? "resume" : "pause" });
};
$("end").onclick = () => send({ type: "end_round" });
document.addEventListener("visibilitychange", () => {
  if (document.hidden && status?.running && !status.paused) {
    input.releaseAll();
    send({ type: "pause" });
  }
});
window.addEventListener("blur", () => send({ type: "input", command: [0, 0, 0, 0, 0, 0, input.gripClosed ? 1 : -1], buttons: {} }));

let last = performance.now();
setInterval(() => {
  const now = performance.now();
  const dt = Math.min(0.2, (now - last) / 1000);
  last = now;
  if (!input.enabled) return;
  send({ type: "input", ...input.sample(dt) });
}, TICK_MS);

connect();
