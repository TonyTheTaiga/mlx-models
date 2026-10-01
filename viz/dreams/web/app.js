// ten dreams — a field of forty: every dream as a point cloud, a column per class, a row per seed.
//
// Dreams arrive as they load, each developing out of its own fog. Drag to look around, scroll to
// come closer; click a dream to fly up to it and watch it develop again from the recorded frames.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Cloud, emergence, fitPoints, shared } from "./depth.js";

const $ = (id) => document.getElementById(id);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const ARRIVE = 3500, REPLAY = 9000, FLY = 1400; // ms

const manifest = await fetch("assets/manifest.json").then((r) => r.json());
// The about text names the model that dreamed these, from the manifest.
if (manifest.model) {
  const NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
  const m = manifest.model, millions = (manifest.params / 1e6).toFixed(1);
  const line = $("modelline").firstChild;
  line.textContent = line.textContent.replace("The model is a vision transformer trained on",
    `The model is a vision transformer — ${NUM[m.n_layers] ?? m.n_layers} blocks, ${NUM[m.n_heads] ?? m.n_heads} heads, about ${millions} million parameters — trained on`);
}

// ----- the scene -----
const canvas = $("field");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.setClearColor(0x050505, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 200);
const controls = new OrbitControls(camera, canvas);
Object.assign(controls, {
  enableDamping: true, dampingFactor: 0.08, rotateSpeed: 0.5, zoomSpeed: 0.8,
  minDistance: 2.5, maxDistance: 70, minAzimuthAngle: -1.2, maxAzimuthAngle: 1.2,
  minPolarAngle: 0.45, maxPolarAngle: Math.PI - 0.45,
});

// Vehicles on the left, animals on the right; seeds top to bottom. The field bows gently
// toward the viewer so the outer columns still face in.
const ORDER = ["airplane", "ship", "automobile", "truck", "bird", "frog", "deer", "horse", "dog", "cat"];
const classes = ORDER.filter((c) => manifest.classes.includes(c)).concat(manifest.classes.filter((c) => !ORDER.includes(c)));
const seeds = Math.max(...manifest.dreams.map((d) => d.seed)) + 1;
const GX = 2.5, GY = 2.4, BOW = 0.012, VIEW_Z = 34;
function place(col, row) {
  const x = (col - (classes.length - 1) / 2) * GX, y = ((seeds - 1) / 2 - row) * GY, z = -BOW * x * x;
  return { pos: new THREE.Vector3(x, y, z), yaw: Math.atan2(-x, VIEW_Z - z) };
}

const slots = manifest.dreams.map((dream) => {
  const col = classes.indexOf(dream.class), row = dream.seed, { pos, yaw } = place(col, row);
  const anchor = new THREE.Group();
  anchor.position.copy(pos);
  anchor.rotation.y = yaw;
  scene.add(anchor);
  anchor.updateMatrixWorld();
  const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(anchor.quaternion);
  return { dream, col, row, anchor, normal, cloud: null, arrived: 0, exposure: 1, replayFrom: 0 };
});
const at = (col, row) => slots.find((s) => s.col === col && s.row === row);

// Class names under each column, kept in place by projecting each frame.
const labels = classes.map((c, col) => {
  const el = Object.assign(document.createElement("div"), { className: "label", textContent: c });
  $("labels").append(el);
  const { pos } = place(col, seeds - 1);
  return { el, world: pos.clone().add(new THREE.Vector3(0, -1.4, 0)) };
});

// ----- the camera: an overview of the field, or close in front of one dream -----
const overview = { pos: new THREE.Vector3(), target: new THREE.Vector3(0, -0.2, -1.5) };
function fitOverview() {
  const w = classes.length * GX + 0.5, h = seeds * GY + 1.8, t = Math.tan((camera.fov * Math.PI) / 360);
  // A little to the left and above, so even the overview shows depth.
  const d = Math.max(h / 2 / t, w / 2 / t / camera.aspect);
  overview.pos.set(-0.14 * d, 0.1 * d, d * 0.99);
}
let flight = null;
function fly(pos, target) {
  if (reduced) { camera.position.copy(pos); controls.target.copy(target); return; }
  flight = { t0: performance.now(), dur: FLY, p0: camera.position.clone(), q0: controls.target.clone(), pos: pos.clone(), target: target.clone() };
}

// ----- frames -----
function frameUrls(dream) {
  return Array.from({ length: dream.frames }, (_, i) => `assets/${dream.folder}/${String(i).padStart(2, "0")}.jpg`);
}
function loadImages(urls) {
  return Promise.all(urls.map((src) => new Promise((ok) => {
    const img = new Image();
    img.onload = img.onerror = () => ok(img); // load events: decode() can stall while the page is hidden
    img.src = src;
  })));
}

let focus = null, hover = null;
async function focusOn(slot) {
  if (!slot?.cloud) return;
  if (focus && focus !== slot) { focus.cloud.setFrames(null); focus.replayFrom = 0; }
  focus = slot;
  // Slightly off-axis, so the relief reads.
  const side = new THREE.Vector3(1, 0, 0).applyQuaternion(slot.anchor.quaternion);
  fly(slot.anchor.position.clone().addScaledVector(slot.normal, 4.1).addScaledVector(side, 1.3).add(new THREE.Vector3(0, 0.6, 0)), slot.anchor.position);
  describe();
  // Develop it again from the recorded frames.
  const frames = await loadImages(frameUrls(slot.dream));
  if (focus !== slot) return;
  slot.cloud.setFrames(frames);
  slot.replayFrom = performance.now() + (reduced ? -REPLAY : FLY * 0.6);
}
function unfocus() {
  if (!focus) return;
  focus.cloud.setFrames(null);
  focus.replayFrom = 0;
  focus = null;
  fly(overview.pos, overview.target);
  describe();
}

// ----- picking: the nearest dream whose square the pointer's ray passes through -----
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), plane = new THREE.Plane(), hit = new THREE.Vector3();
function pick(e) {
  ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  let best = null, bestD = Infinity;
  for (const s of slots) {
    if (!s.cloud) continue;
    plane.setFromNormalAndCoplanarPoint(s.normal, s.anchor.position);
    if (!ray.ray.intersectPlane(plane, hit)) continue;
    const d = hit.distanceTo(ray.ray.origin), local = s.anchor.worldToLocal(hit.clone());
    if (Math.abs(local.x) < 1.05 && Math.abs(local.y) < 1.05 && d < bestD) { best = s; bestD = d; }
  }
  return best;
}

const info = $("info");
function describe() {
  const s = focus || hover;
  canvas.style.cursor = hover ? "pointer" : "";
  info.classList.toggle("on", !!s);
  if (!s) return;
  const a = s.dream.class === "deer" ? "a deer" : `${/^[aeiou]/.test(s.dream.class) ? "an" : "a"} ${s.dream.class}`;
  info.innerHTML = `<i>${a}</i><span>dream ${s.row + 1} of ${seeds} · ${s.dream.belief > 0.9995 ? "over 99.9" : (s.dream.belief * 100).toFixed(1)}% certain</span>`;
}

// ----- input: click a dream to go to it, click the dark to step back -----
let down = null;
canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; flight = null; });
canvas.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return; // a drag, not a click
  const s = pick(e);
  if (s) focusOn(s); // the focused one again replays it
  else unfocus();
});
canvas.addEventListener("pointermove", (e) => {
  if (e.buttons) return;
  const s = pick(e);
  if (s !== hover) { hover = s; describe(); }
});
canvas.addEventListener("pointerleave", () => { hover = null; describe(); });
const about = $("about");
addEventListener("keydown", (e) => {
  if (e.key === "i" || (e.key === "Escape" && !about.hidden)) { about.hidden = !about.hidden; return; }
  if (e.key === "Escape") return unfocus();
  const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
  if (!step) return;
  e.preventDefault();
  const from = focus || hover;
  if (!from) return focusOn(at(0, 0));
  focusOn(at((from.col + step[0] + classes.length) % classes.length, (from.row + step[1] + seeds) % seeds));
});
about.addEventListener("click", () => (about.hidden = true));

// ----- the loop -----
function resize() {
  const atOverview = !focus && !flight && camera.position.distanceTo(overview.pos) < 0.01;
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  fitPoints(camera, innerHeight * renderer.getPixelRatio());
  fitOverview();
  if (atOverview) { camera.position.copy(overview.pos); controls.target.copy(overview.target); }
}
addEventListener("resize", resize);
resize();
// Begin a little further back and drift in while the field fills.
camera.position.copy(overview.pos).add(new THREE.Vector3(0, -2, 10));
controls.target.copy(overview.target);
if (!reduced) flight = { t0: performance.now(), dur: 4000, p0: camera.position.clone(), q0: overview.target.clone(), pos: overview.pos.clone(), target: overview.target.clone() };
else camera.position.copy(overview.pos);

const ease = (t) => t * t * (3 - 2 * t);
const v = new THREE.Vector3();
function frame(now) {
  requestAnimationFrame(frame);
  shared.time.value = now / 1000;
  if (flight) {
    const k = ease(Math.min(1, (now - flight.t0) / flight.dur));
    camera.position.lerpVectors(flight.p0, flight.pos, k);
    controls.target.lerpVectors(flight.q0, flight.target, k);
    if (k >= 1) flight = null;
  }
  controls.update();
  for (const s of slots) {
    if (!s.cloud) continue;
    let dev = reduced ? 1 : ease(Math.min(1, (now - s.arrived) / ARRIVE));
    if (s === focus && s.replayFrom) dev = Math.min(1, Math.max(0, (now - s.replayFrom) / REPLAY));
    // When one dream is chosen the others recede; the one under the pointer brightens.
    const want = focus ? (s === focus ? 1 : 0.2) : s === hover ? 1.2 : 1;
    s.exposure += (want - s.exposure) * 0.08;
    s.cloud.set(dev, s.exposure);
  }
  for (const l of labels) {
    v.copy(l.world).project(camera);
    const visible = v.z < 1 && Math.abs(v.x) < 1.1 && Math.abs(v.y) < 1.1;
    l.el.style.opacity = visible ? (focus ? 0.2 : 0.7) : 0;
    l.el.style.transform = `translate(${((v.x + 1) / 2) * innerWidth}px, ${((1 - v.y) / 2) * innerHeight}px) translate(-50%, -50%)`;
  }
  renderer.render(scene, camera);
}
requestAnimationFrame(frame);

// ----- loading: each dream's frames once, to find its depth; afterwards only the final frame is kept.
// A few at a time in shuffled order, so the field fills in like prints coming up in a tray. -----
const count = $("count"), pending = [...slots].sort(() => Math.random() - 0.5);
let loaded = 0;
async function arrive(slot) {
  const frames = await loadImages(frameUrls(slot.dream));
  slot.cloud = new Cloud(emergence(frames), frames[frames.length - 1]);
  slot.anchor.add(slot.cloud.points);
  slot.arrived = performance.now();
  count.textContent = `${++loaded} of ${slots.length} dreams`;
}
Promise.all(Array.from({ length: 4 }, async () => {
  while (pending.length) await arrive(pending.shift());
})).then(() => count.classList.add("done"));
