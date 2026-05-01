import { add, average, clonePoints, mulberry32, randomNormal, squaredDist, vec } from "./math.js";

const defaultCloudBounds = {
  minX: -2.4,
  maxX: 2.4,
  minY: -2.1,
  maxY: 2.1,
};

function seededPointCloud(seed = 12) {
  const rng = mulberry32(seed);
  const points = [];

  function pushCluster(center, count, sx, sy) {
    for (let i = 0; i < count; i += 1) {
      points.push(vec(center.x + randomNormal(rng) * sx, center.y + randomNormal(rng) * sy));
    }
  }

  pushCluster(vec(0, 0), 320, 0.24, 0.19);
  pushCluster(vec(-1.45, 0.9), 135, 0.19, 0.13);
  pushCluster(vec(-1.2, -0.95), 128, 0.18, 0.14);
  pushCluster(vec(-0.2, 1.4), 114, 0.17, 0.12);
  pushCluster(vec(1.18, 1.08), 138, 0.19, 0.14);
  pushCluster(vec(1.46, 0.28), 132, 0.16, 0.11);
  pushCluster(vec(1.12, -0.85), 126, 0.17, 0.13);
  pushCluster(vec(-0.1, -1.52), 120, 0.17, 0.16);

  for (let i = 0; i < 240; i += 1) {
    const t = rng() * 4.2 - 2.1;
    points.push(vec(0.98 * t + randomNormal(rng) * 0.08, 0.5 * t + randomNormal(rng) * 0.09));
  }
  for (let i = 0; i < 210; i += 1) {
    const t = rng() * 3.8 - 1.9;
    points.push(vec(0.48 * t + randomNormal(rng) * 0.08, -1.02 * t + randomNormal(rng) * 0.08));
  }

  return points;
}

function randomSeed() {
  return Math.floor(Math.random() * 4294967296);
}

function uniformPointCloud(count, seed = 12, bounds = defaultCloudBounds) {
  const rng = mulberry32(seed);
  const points = [];
  for (let i = 0; i < count; i += 1) {
    points.push(
      vec(
        bounds.minX + rng() * (bounds.maxX - bounds.minX),
        bounds.minY + rng() * (bounds.maxY - bounds.minY),
      ),
    );
  }
  return points;
}

function farthestPointSampling(points, count, seed = 77) {
  const rng = mulberry32(seed + count * 17);
  const first = Math.floor(rng() * points.length);
  const chosen = [points[first]];
  const used = new Set([first]);

  while (chosen.length < count) {
    let bestIndex = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < points.length; i += 1) {
      if (used.has(i)) {
        continue;
      }
      let minD2 = Infinity;
      for (const center of chosen) {
        minD2 = Math.min(minD2, squaredDist(points[i], center));
      }
      if (minD2 > bestScore) {
        bestScore = minD2;
        bestIndex = i;
      }
    }
    if (bestIndex === -1) {
      break;
    }
    used.add(bestIndex);
    chosen.push(points[bestIndex]);
  }

  return clonePoints(chosen.slice(0, count));
}

function sampleInitialHubs(points, count, seed = 77) {
  return farthestPointSampling(points, count, seed).map((point, index) =>
    add(point, vec(((index % 3) - 1) * 0.03, (((index + 1) % 3) - 1) * 0.03)),
  );
}

function nearestCenterIndex(point, centers) {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < centers.length; i += 1) {
    const d2 = squaredDist(point, centers[i]);
    if (d2 < bestDist) {
      bestDist = d2;
      best = i;
    }
  }
  return best;
}

function kMeans(points, k, iterations = 12, seed = 91, initialCenters = null) {
  if (points.length === 0 || k <= 0) {
    return { centers: [], assignments: [] };
  }
  const centerCount = Math.min(k, points.length);
  let centers = initialCenters ? clonePoints(initialCenters.slice(0, centerCount)) : sampleInitialHubs(points, centerCount, seed);
  const assignments = Array(points.length).fill(0);

  for (let iter = 0; iter < iterations; iter += 1) {
    for (let i = 0; i < points.length; i += 1) {
      assignments[i] = nearestCenterIndex(points[i], centers);
    }

    const buckets = Array.from({ length: centerCount }, () => []);
    for (let i = 0; i < points.length; i += 1) {
      buckets[assignments[i]].push(points[i]);
    }

    centers = centers.map((center, index) => (buckets[index].length > 0 ? average(buckets[index]) : center));
  }

  return { centers, assignments };
}

export {
  defaultCloudBounds,
  farthestPointSampling,
  kMeans,
  nearestCenterIndex,
  randomSeed,
  sampleInitialHubs,
  seededPointCloud,
  uniformPointCloud,
};
