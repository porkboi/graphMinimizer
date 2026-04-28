const TAU = Math.PI * 2;

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let value = Math.imul(t ^ (t >>> 15), t | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNormal(rng) {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TAU * u2);
}

function vec(x = 0, y = 0) {
  return { x, y };
}

function add(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}

function scale(a, s) {
  return { x: a.x * s, y: a.y * s };
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function squaredDist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function norm(a) {
  return Math.hypot(a.x, a.y);
}

function normalize(a) {
  const n = norm(a);
  if (n < 1e-9) {
    return vec(0, 0);
  }
  return scale(a, 1 / n);
}

function average(points) {
  if (points.length === 0) {
    return vec(0, 0);
  }
  let sum = vec(0, 0);
  for (const point of points) {
    sum = add(sum, point);
  }
  return scale(sum, 1 / points.length);
}

function clonePoints(points) {
  return points.map((point) => ({ x: point.x, y: point.y }));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export {
  TAU,
  add,
  average,
  clamp,
  clonePoints,
  dist,
  mulberry32,
  norm,
  normalize,
  randomNormal,
  scale,
  squaredDist,
  sub,
  vec,
};
