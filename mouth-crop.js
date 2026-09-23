const REFERENCE = [
  [102.074, 94.272],
  [156.361, 93.578],
  [129.004, 135.903],
  [129.313, 157.823],
];
const SIZE = 96;

const RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153];
const LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380];
const NOSE_TIP = [1];
const MOUTH = [13, 14, 0, 17, 61, 291];

const canvas = document.createElement('canvas');
canvas.width = SIZE;
canvas.height = SIZE;
const ctx = canvas.getContext('2d', { willReadFrequently: true });

export function cropMouth(video, landmarks) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const mean = (ids) => {
    let x = 0;
    let y = 0;
    for (const i of ids) {
      x += landmarks[i].x * w;
      y += landmarks[i].y * h;
    }
    return [x / ids.length, y / ids.length];
  };
  const src = [mean(RIGHT_EYE), mean(LEFT_EYE), mean(NOSE_TIP), mean(MOUTH)];

  // Least-squares similarity transform (rotation + uniform scale + translation) src -> REFERENCE.
  const cs = centroid(src);
  const cd = centroid(REFERENCE);
  let num1 = 0;
  let num2 = 0;
  let den = 0;
  for (let i = 0; i < 4; i++) {
    const sx = src[i][0] - cs[0];
    const sy = src[i][1] - cs[1];
    const dx = REFERENCE[i][0] - cd[0];
    const dy = REFERENCE[i][1] - cd[1];
    num1 += sx * dx + sy * dy;
    num2 += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }
  if (den < 1e-6) return null;
  const a = num1 / den;
  const b = num2 / den;
  const tx = cd[0] - (a * cs[0] - b * cs[1]);
  const ty = cd[1] - (b * cs[0] + a * cs[1]);

  // Centre of the crop = the mouth centre after alignment.
  const [mx, my] = src[3];
  const cx = a * mx - b * my + tx;
  const cy = b * mx + a * my + ty;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.setTransform(a, b, -b, a, tx - (cx - SIZE / 2), ty - (cy - SIZE / 2));
  ctx.drawImage(video, 0, 0);

  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const gray = new Uint8Array(SIZE * SIZE);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = Math.round(0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]);
  }
  return gray;
}

function centroid(pts) {
  let x = 0;
  let y = 0;
  for (const [px, py] of pts) {
    x += px;
    y += py;
  }
  return [x / pts.length, y / pts.length];
}

export function packFrames(frames) {
  const bytes = new Uint8Array(frames.length * SIZE * SIZE);
  frames.forEach((f, i) => bytes.set(f, i * SIZE * SIZE));
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

// Last crop, for the small "what the model sees" preview.
export function lastCropCanvas() {
  return canvas;
}
