// The dreams in depth: every pixel a point, placed in space by when the model decided it.
//
// Depth comes from the recorded optimization, not from the image: for each pixel, the fraction of
// the dream during which it was still far from its final color. Pixels the model settled early sit
// at the back; the last details to resolve come nearest. During playback each point flies in out
// of the fog at the moment its pixel settles, colored by the developing frames.
import * as THREE from "three";

const SIZE = 256;

// Per-pixel settling time in [0, 1], on the same frame timeline as playback.
export function emergence(frames) {
  const c = Object.assign(document.createElement("canvas"), { width: SIZE, height: SIZE });
  const g = c.getContext("2d", { willReadFrequently: true });
  const px = frames.map((img) => { g.drawImage(img, 0, 0, SIZE, SIZE); return g.getImageData(0, 0, SIZE, SIZE).data; });
  const n = px.length, last = px[n - 1], N = SIZE * SIZE, tau = 0.1 * 255 * Math.sqrt(3);
  let e = new Float32Array(N);
  for (let p = 0; p < N; p++) {
    let acc = 0;
    for (let i = 0; i < n - 1; i++) {
      const o = p * 4, dr = px[i][o] - last[o], dg = px[i][o + 1] - last[o + 1], db = px[i][o + 2] - last[o + 2];
      acc += Math.min(1, Math.hypot(dr, dg, db) / tau);
    }
    e[p] = acc / (n - 1);
  }
  // Soften the terraces from the 24-frame sampling, then spread to the full range.
  for (let pass = 0; pass < 2; pass++) e = blur(e);
  const sorted = Float32Array.from(e).sort(), lo = sorted[Math.floor(N * 0.02)], hi = sorted[Math.floor(N * 0.98)];
  for (let p = 0; p < N; p++) e[p] = Math.min(1, Math.max(0, (e[p] - lo) / (hi - lo || 1)));
  return e;
}
function blur(a) {
  const out = new Float32Array(a.length), r = 3;
  for (let y = 0; y < SIZE; y++)
    for (let x = 0; x < SIZE; x++) {
      let s = 0, k = 0;
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || xx < 0 || yy >= SIZE || xx >= SIZE) continue;
          s += a[yy * SIZE + xx]; k++;
        }
      out[y * SIZE + x] = s / k;
    }
  return out;
}

const vertex = /* glsl */ `
  attribute float settle;   // when this pixel was decided, 0..1
  attribute vec3 seed;      // where it waits in the fog
  uniform sampler2D texA, texB;
  uniform float mixAB, dev, time, depth, pointScale;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    // A point arrives when development reaches the moment its pixel settled.
    float s = smoothstep(settle - 0.08, settle + 0.06, dev);
    vec3 target = vec3((uv.x - 0.5) * 2.0, (uv.y - 0.5) * 2.0, (settle - 0.4) * depth);
    target.z += 0.012 * sin(time * 0.6 + uv.x * 7.0 + uv.y * 5.0);       // breathing
    vec3 fog = target + seed * 0.9 + vec3(0.0, 0.0, -1.6);                 // behind, scattered
    vec3 p = mix(fog, target, s);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = pointScale * (1.0 + 1.6 * (1.0 - s)) / -mv.z;
    vColor = mix(texture2D(texA, uv).rgb, texture2D(texB, uv).rgb, mixAB);
    vAlpha = mix(0.18, 1.0, s);
  }`;
const fragment = /* glsl */ `
  uniform float exposure;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = dot(c, c);
    if (r > 0.25) discard;
    float soft = smoothstep(0.25, 0.12, r);
    gl_FragColor = vec4(vColor * exposure, vAlpha * soft);
  }`;

export function createDepth(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(0x050505, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);

  const N = SIZE * SIZE, pos = new Float32Array(N * 3), uv = new Float32Array(N * 2), seed = new Float32Array(N * 3);
  for (let p = 0; p < N; p++) {
    const x = p % SIZE, y = Math.floor(p / SIZE);
    uv[p * 2] = (x + 0.5) / SIZE;
    uv[p * 2 + 1] = 1 - (y + 0.5) / SIZE;
    for (let k = 0; k < 3; k++) seed[p * 3 + k] = (Math.random() - 0.5) * 2;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3)); // computed in the shader
  geom.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geom.setAttribute("seed", new THREE.BufferAttribute(seed, 3));
  const settleAttr = new THREE.BufferAttribute(new Float32Array(N), 1);
  geom.setAttribute("settle", settleAttr);
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 4);

  const uniforms = {
    texA: { value: null }, texB: { value: null }, mixAB: { value: 0 }, dev: { value: 0 }, time: { value: 0 },
    depth: { value: 1.1 }, pointScale: { value: 10 }, exposure: { value: 1 },
  };
  const material = new THREE.ShaderMaterial({ vertexShader: vertex, fragmentShader: fragment, uniforms, transparent: true, depthWrite: false });
  scene.add(new THREE.Points(geom, material));

  let textures = [];
  function setDream(frames, settle) {
    textures.forEach((t) => t.dispose());
    textures = frames.map((img) => {
      const t = new THREE.Texture(img);
      t.colorSpace = THREE.SRGBColorSpace;
      t.minFilter = THREE.LinearFilter;
      t.generateMipmaps = false;
      t.needsUpdate = true;
      return t;
    });
    settleAttr.array.set(settle);
    settleAttr.needsUpdate = true;
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio())) renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // Points a little wider than the pixel pitch, so the settled image closes up.
    const perUnit = (h * renderer.getPixelRatio()) / (2 * Math.tan((camera.fov * Math.PI) / 360));
    uniforms.pointScale.value = perUnit * (2 / SIZE) * 1.5;
  }

  // t: seconds into this dream; dev: development 0..1; exposure: fade.
  function render({ dev, frame, frac, t, exposure, still }) {
    resize();
    uniforms.texA.value = textures[frame];
    uniforms.texB.value = textures[Math.min(frame + 1, textures.length - 1)];
    uniforms.mixAB.value = frac;
    uniforms.dev.value = dev;
    uniforms.time.value = t;
    uniforms.exposure.value = exposure;
    // A slow orbit that widens as the dream resolves, so the depth reads; still for reduced motion.
    const swing = still ? 0 : 0.12 + 0.3 * dev;
    const yaw = swing * Math.sin(t * 0.11), pitch = (still ? 0 : 0.08) * Math.sin(t * 0.07 + 1);
    const dist = 5 - (still ? 0 : 0.35 * dev);
    camera.position.set(dist * Math.sin(yaw) * Math.cos(pitch), dist * Math.sin(pitch), dist * Math.cos(yaw) * Math.cos(pitch));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }

  return { setDream, render };
}
