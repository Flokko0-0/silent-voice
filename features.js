// Outer and inner lip contours, ordered around the mouth (MediaPipe 468-point mesh indices).
export const OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
export const INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

const EYE_A = 33;
const EYE_B = 263;
const NOSE_TIP = 1;
const CHIN = 152;

// Blendshapes that describe the lower face; eye/brow blendshapes are noise for lip reading.
const MOUTH_BLENDSHAPES = [
  'jawOpen', 'jawLeft', 'jawRight', 'jawForward', 'mouthClose', 'mouthFunnel', 'mouthPucker',
  'mouthLeft', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthRollLower',
  'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight',
  'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'cheekPuff',
];

const GEOM_COUNT = 11;
const CONTOUR_COUNT = (OUTER_LIPS.length + INNER_LIPS.length) * 2;

export const FEATURE_GROUPS = [
  ...Array(GEOM_COUNT).fill(0),
  ...Array(MOUTH_BLENDSHAPES.length).fill(1),
  ...Array(CONTOUR_COUNT).fill(2),
];
export const FEATURE_DIM = FEATURE_GROUPS.length;

export const ACTIVITY_FLOOR = FEATURE_GROUPS.map((g) => (g === 0 ? 0.012 : g === 1 ? 0.04 : Infinity));

export function extractFeatures(landmarks, blendshapes, width, height) {
  const px = (i) => ({ x: landmarks[i].x * width, y: landmarks[i].y * height, z: landmarks[i].z * width });

  const a = px(EYE_A);
  const b = px(EYE_B);
  const scale = Math.hypot(b.x - a.x, b.y - a.y);
  if (scale < 1) return null;
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  const left = px(61);
  const right = px(291);
  const cx = (left.x + right.x) / 2;
  const cy = (left.y + right.y) / 2;
  const cz = (left.z + right.z) / 2;

  const norm = (i) => {
    const p = px(i);
    const dx = (p.x - cx) / scale;
    const dy = (p.y - cy) / scale;
    return { x: dx * cos - dy * sin, y: dx * sin + dy * cos, z: (p.z - cz) / scale };
  };
  const d = (i, j) => {
    const p = norm(i);
    const q = norm(j);
    return Math.hypot(p.x - q.x, p.y - q.y);
  };

  const protrusion = (norm(0).z + norm(17).z + norm(13).z + norm(14).z) / 4 - (norm(61).z + norm(291).z) / 2;
  const geom = [
    d(61, 291), // mouth width
    d(13, 14), // inner opening (centre)
    d(0, 17), // outer height
    d(82, 87), // inner opening (left)
    d(312, 317), // inner opening (right)
    d(37, 84), // outer height (left)
    d(267, 314), // outer height (right)
    d(0, 13), // upper lip thickness (rolls in on "м/б/п")
    d(14, 17), // lower lip thickness
    d(NOSE_TIP, CHIN), // jaw drop
    protrusion, // lips pushed forward ("у/о")
  ];

  const scores = {};
  for (const c of blendshapes || []) scores[c.categoryName] = c.score;
  const bs = MOUTH_BLENDSHAPES.map((name) => scores[name] ?? 0);

  const contour = [];
  for (const i of [...OUTER_LIPS, ...INNER_LIPS]) {
    const p = norm(i);
    contour.push(p.x, p.y);
  }

  const toPixels = (ids) => ids.map((i) => [landmarks[i].x * width, landmarks[i].y * height]);
  return {
    vec: [...geom, ...bs, ...contour],
    outer: toPixels(OUTER_LIPS),
    inner: toPixels(INNER_LIPS),
  };
}
