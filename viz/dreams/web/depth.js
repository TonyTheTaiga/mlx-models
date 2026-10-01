// The dreams in depth: every pixel a point, placed in space by when the model decided it.
//
// Depth comes from the recorded optimization, not from the image: for each pixel, the fraction of
// the dream during which it was still far from its final color. Pixels the model settled early sit
// at the back; the last details to resolve come nearest. A cloud develops by flying each point in
// out of the fog at the moment its pixel settled, colored by the developing frames.
import * as THREE from "three";

export const SIZE = 256;

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

export function texture(img) {
  const t = new THREE.Texture(img);
  t.colorSpace = THREE.SRGBColorSpace;
  t.minFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
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

// Uniforms every cloud shares: the clock, the relief, and the point size for the viewport.
export const shared = { time: { value: 0 }, pointScale: { value: 10 }, depth: { value: 1.1 } };
// Points a little wider than the pixel pitch, so a settled cloud closes up at any distance.
export function fitPoints(camera, heightPx) {
  const perUnit = heightPx / (2 * Math.tan((camera.fov * Math.PI) / 360));
  shared.pointScale.value = perUnit * (2 / SIZE) * 1.5;
}

let common = null; // uv, fog seeds and a placeholder position: the same for every cloud
function attributes() {
  if (common) return common;
  const N = SIZE * SIZE, uv = new Float32Array(N * 2), seed = new Float32Array(N * 3);
  for (let p = 0; p < N; p++) {
    uv[p * 2] = ((p % SIZE) + 0.5) / SIZE;
    uv[p * 2 + 1] = 1 - (Math.floor(p / SIZE) + 0.5) / SIZE;
    for (let k = 0; k < 3; k++) seed[p * 3 + k] = (Math.random() - 0.5) * 2;
  }
  common = {
    position: new THREE.BufferAttribute(new Float32Array(N * 3), 3), // computed in the shader
    uv: new THREE.BufferAttribute(uv, 2),
    seed: new THREE.BufferAttribute(seed, 3),
  };
  return common;
}

// One dream as a point cloud: a 2×2 square in its local xy plane, facing +z.
export class Cloud {
  constructor(settle, final) {
    const geom = new THREE.BufferGeometry(), a = attributes();
    for (const k in a) geom.setAttribute(k, a[k]);
    geom.setAttribute("settle", new THREE.BufferAttribute(settle, 1));
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 2.5);
    this.final = texture(final);
    this.uniforms = {
      ...shared, texA: { value: this.final }, texB: { value: this.final },
      mixAB: { value: 0 }, dev: { value: 0 }, exposure: { value: 1 },
    };
    this.points = new THREE.Points(geom, new THREE.ShaderMaterial({
      vertexShader: vertex, fragmentShader: fragment, uniforms: this.uniforms, transparent: true, depthWrite: false,
    }));
    this.frames = null;
  }
  // The recorded frames, for replaying development; null to keep only the final image.
  setFrames(images) {
    this.frames?.forEach((t) => t.dispose());
    this.frames = images && images.map(texture);
  }
  set(dev, exposure) {
    const u = this.uniforms;
    u.dev.value = dev;
    u.exposure.value = exposure;
    if (!this.frames) { u.texA.value = u.texB.value = this.final; return; }
    const n = this.frames.length, x = Math.min(1, dev) * (n - 1), i = Math.min(n - 2, Math.floor(x));
    u.texA.value = this.frames[i];
    u.texB.value = this.frames[i + 1];
    u.mixAB.value = x - i;
  }
}
