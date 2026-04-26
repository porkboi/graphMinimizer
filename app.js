const TAU = Math.PI * 2;
const sceneCanvas = document.getElementById("scene");
const sceneCtx = sceneCanvas.getContext("2d");
const objectiveCanvas = document.getElementById("objectiveChart");
const objectiveCtx = objectiveCanvas.getContext("2d");
const residualCanvas = document.getElementById("residualChart");
const residualCtx = residualCanvas.getContext("2d");

const ui = {
  iterValue: document.getElementById("iterValue"),
  objectiveValue: document.getElementById("objectiveValue"),
  primalValue: document.getElementById("primalValue"),
  dualValue: document.getElementById("dualValue"),
  fxValue: document.getElementById("fxValue"),
  g1Value: document.getElementById("g1Value"),
  g2Value: document.getElementById("g2Value"),
  g3Value: document.getElementById("g3Value"),
  g4Value: document.getElementById("g4Value"),
  weightLabel: document.getElementById("weightLabel"),
  weightValue: document.getElementById("weightValue"),
  pathLabel: document.getElementById("pathLabel"),
  pathValue: document.getElementById("pathValue"),
  playPause: document.getElementById("playPause"),
  stepButton: document.getElementById("stepButton"),
  resetButton: document.getElementById("resetButton"),
  randomizeButton: document.getElementById("randomizeButton"),
  voronoiToggle: document.getElementById("voronoiToggle"),
  canvasSolverGroup: document.getElementById("canvasSolverGroup"),
  canvasExplorerMode: document.getElementById("canvasExplorerMode"),
  canvasTuningMode: document.getElementById("canvasTuningMode"),
  hubCountGroup: document.getElementById("hubCountGroup"),
  hubCount: document.getElementById("hubCount"),
  hubCountValue: document.getElementById("hubCountValue"),
  rho: document.getElementById("rho"),
  rhoValue: document.getElementById("rhoValue"),
  lambda: document.getElementById("lambda"),
  lambdaValue: document.getElementById("lambdaValue"),
  mu: document.getElementById("mu"),
  muValue: document.getElementById("muValue"),
  stationCost: document.getElementById("stationCost"),
  stationCostValue: document.getElementById("stationCostValue"),
  explorerTab: document.getElementById("explorerTab"),
  tuningTab: document.getElementById("tuningTab"),
  canvasTab: document.getElementById("canvasTab"),
  modeSummary: document.getElementById("modeSummary"),
  zoomIn: document.getElementById("zoomIn"),
  zoomOut: document.getElementById("zoomOut"),
  panUp: document.getElementById("panUp"),
  panDown: document.getElementById("panDown"),
  panLeft: document.getElementById("panLeft"),
  panRight: document.getElementById("panRight"),
};

const modeCopy = {
  explorer:
    'The browser runs a 2D, deterministic ADMM-style solver over hub locations <code>x, z₁, z₂, z₃, u₁, u₂, u₃</code>. Each frame updates point assignments, shortest-path regularisation, quadratic shrinkage, graph total variation, and a station-count penalty.',
  tuning:
    "This tab reuses the explorer's local kNN-graph ADMM solver at each fixed hub count. It starts from 3 KMeans hubs, settles the current geometry with the same ADMM updates as the explorer, then splits the highest-energy Voronoi cell and keeps the new hub only when that locally converged split lowers the loss after the station penalty is included.",
  canvas:
    "This tab turns the main canvas into a point-cloud editor. Each click adds an observed point, then the browser re-seeds hub locations from that clicked cloud and solves with the selected <code>app.js</code> path: the explorer's local kNN-graph ADMM update or the existing hub-tuning loop.",
};

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

function edgeKey(a, b) {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return `${lo}-${hi}`;
}

function canonicalizeEdges(edges) {
  const unique = new Set();
  const canonical = [];
  for (const [aRaw, bRaw] of edges) {
    const a = Math.min(aRaw, bRaw);
    const b = Math.max(aRaw, bRaw);
    if (a === b) {
      continue;
    }
    const key = `${a}-${b}`;
    if (!unique.has(key)) {
      unique.add(key);
      canonical.push([a, b]);
    }
  }
  return canonical;
}

function graphFromEdges(points, edges) {
  const adjacency = points.map(() => []);
  for (const [a, b] of canonicalizeEdges(edges)) {
    const w = dist(points[a], points[b]);
    adjacency[a].push({ to: b, w });
    adjacency[b].push({ to: a, w });
  }
  return { adjacency, edges: canonicalizeEdges(edges) };
}

function buildKnnGraph(points, k = 2) {
  const edgeSet = new Set();
  const edges = [];

  for (let i = 0; i < points.length; i += 1) {
    const neighbors = [];
    for (let j = 0; j < points.length; j += 1) {
      if (i === j) {
        continue;
      }
      neighbors.push({ j, d: dist(points[i], points[j]) });
    }
    neighbors.sort((a, b) => a.d - b.d);
    for (const { j } of neighbors.slice(0, Math.min(k, neighbors.length))) {
      const key = edgeKey(i, j);
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push(key.split("-").map(Number));
      }
    }
  }

  return graphFromEdges(points, edges);
}

function buildCompleteGraph(points) {
  const edges = [];
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      edges.push([i, j]);
    }
  }
  return graphFromEdges(points, edges);
}

function buildMaxGraph(points) {
  return buildCompleteGraph(points);
}

function dijkstra(adjacency, start) {
  const n = adjacency.length;
  const distArr = Array(n).fill(Infinity);
  const prev = Array(n).fill(-1);
  const visited = Array(n).fill(false);
  distArr[start] = 0;

  for (let step = 0; step < n; step += 1) {
    let u = -1;
    let best = Infinity;
    for (let i = 0; i < n; i += 1) {
      if (!visited[i] && distArr[i] < best) {
        best = distArr[i];
        u = i;
      }
    }
    if (u === -1) {
      break;
    }
    visited[u] = true;
    for (const { to, w } of adjacency[u]) {
      const candidate = distArr[u] + w;
      if (candidate < distArr[to]) {
        distArr[to] = candidate;
        prev[to] = u;
      }
    }
  }

  return { dist: distArr, prev };
}

function reconstructPath(prev, start, end) {
  if (start === end) {
    return [start];
  }
  const path = [];
  let current = end;
  while (current !== -1) {
    path.push(current);
    if (current === start) {
      break;
    }
    current = prev[current];
  }
  path.reverse();
  return path[0] === start ? path : [];
}

function isGraphConnected(points, edges) {
  if (points.length <= 1) {
    return true;
  }
  const graph = graphFromEdges(points, edges);
  const queue = [0];
  const seen = new Set([0]);
  while (queue.length > 0) {
    const node = queue.shift();
    for (const { to } of graph.adjacency[node]) {
      if (!seen.has(to)) {
        seen.add(to);
        queue.push(to);
      }
    }
  }
  return seen.size === points.length;
}

function pathContainsEdge(path, a, b) {
  const key = edgeKey(a, b);
  for (let i = 0; i < path.length - 1; i += 1) {
    if (edgeKey(path[i], path[i + 1]) === key) {
      return true;
    }
  }
  return false;
}

function pathSquaredLength(points, path) {
  let total = 0;
  for (let i = 0; i < path.length - 1; i += 1) {
    total += squaredDist(points[path[i]], points[path[i + 1]]);
  }
  return total;
}

function assignPoints(hubs, points) {
  const assignments = Array.from({ length: hubs.length }, () => []);
  const weights = Array(hubs.length).fill(0);

  for (const point of points) {
    const best = nearestCenterIndex(point, hubs);
    assignments[best].push(point);
    weights[best] += 1;
  }

  return { assignments, weights };
}

function computeFx(hubs, assignments) {
  let total = 0;
  for (let i = 0; i < hubs.length; i += 1) {
    const hub = hubs[i];
    for (const point of assignments[i]) {
      total += 0.5 * squaredDist(point, hub);
    }
  }
  return total;
}

function computeG2(z2, lambda) {
  let sum = 0;
  for (const point of z2) {
    sum += point.x * point.x + point.y * point.y;
  }
  return lambda * sum;
}

function computeGraphVariation(points, edges, mu) {
  let total = 0;
  for (const [a, b] of edges) {
    total += mu * dist(points[a], points[b]);
  }
  return total;
}

function computeStationBuildCost(hubs, stationCost) {
  return hubs.length * stationCost;
}

function computePathSmoothness(points, weights, graph) {
  const n = points.length;
  const totalMass = Math.max(weights.reduce((sum, value) => sum + value, 0), 1);
  let smoothness = 0;
  let highlighted = { pair: [0, 0], path: [0], score: 0 };

  for (let i = 0; i < n; i += 1) {
    const result = dijkstra(graph.adjacency, i);
    for (let j = i + 1; j < n; j += 1) {
      if (!Number.isFinite(result.dist[j])) {
        continue;
      }
      const path = reconstructPath(result.prev, i, j);
      if (path.length === 0) {
        continue;
      }
      let pathPenalty = 0;
      for (let p = 0; p < path.length - 1; p += 1) {
        pathPenalty += dist(points[path[p]], points[path[p + 1]]);
      }
      const score = 0.5 * (weights[i] * weights[j]) / (totalMass * totalMass) * pathPenalty;
      smoothness += score;
      if (score > highlighted.score) {
        highlighted = { pair: [i, j], path, score };
      }
    }
  }

  return { value: smoothness, highlighted };
}

function computeClusterEnergies(assignments, hubs) {
  return assignments.map((bucket, index) => {
    let total = 0;
    for (const point of bucket) {
      total += squaredDist(point, hubs[index]);
    }
    return total;
  });
}

function clipPolygonWithHalfPlane(polygon, signedDistance) {
  if (polygon.length === 0) {
    return [];
  }

  const clipped = [];
  for (let i = 0; i < polygon.length; i += 1) {
    const current = polygon[i];
    const next = polygon[(i + 1) % polygon.length];
    const currentDistance = signedDistance(current);
    const nextDistance = signedDistance(next);
    const currentInside = currentDistance <= 1e-9;
    const nextInside = nextDistance <= 1e-9;

    if (currentInside && nextInside) {
      clipped.push(next);
      continue;
    }

    if (currentInside !== nextInside) {
      const denom = currentDistance - nextDistance;
      const t = Math.abs(denom) < 1e-9 ? 0 : currentDistance / denom;
      clipped.push(vec(current.x + (next.x - current.x) * t, current.y + (next.y - current.y) * t));
    }

    if (!currentInside && nextInside) {
      clipped.push(next);
    }
  }

  return clipped;
}

function computeVoronoiCells(hubs, bounds) {
  if (hubs.length === 0) {
    return [];
  }

  const boundsPolygon = [
    vec(bounds.minX, bounds.minY),
    vec(bounds.maxX, bounds.minY),
    vec(bounds.maxX, bounds.maxY),
    vec(bounds.minX, bounds.maxY),
  ];

  return hubs.map((hub, index) => {
    let polygon = boundsPolygon;
    for (let otherIndex = 0; otherIndex < hubs.length; otherIndex += 1) {
      if (otherIndex === index) {
        continue;
      }
      const other = hubs[otherIndex];
      const dx = other.x - hub.x;
      const dy = other.y - hub.y;
      const offset = 0.5 * (other.x * other.x + other.y * other.y - hub.x * hub.x - hub.y * hub.y);
      polygon = clipPolygonWithHalfPlane(polygon, (point) => dx * point.x + dy * point.y - offset);
      if (polygon.length === 0) {
        break;
      }
    }
    return polygon;
  });
}

function polygonCentroid(polygon) {
  if (polygon.length === 0) {
    return null;
  }
  if (polygon.length < 3) {
    return average(polygon);
  }

  let area = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const current = polygon[i];
    const next = polygon[(i + 1) % polygon.length];
    const cross = current.x * next.y - next.x * current.y;
    area += cross;
    cx += (current.x + next.x) * cross;
    cy += (current.y + next.y) * cross;
  }

  if (Math.abs(area) < 1e-9) {
    return average(polygon);
  }

  return vec(cx / (3 * area), cy / (3 * area));
}

function proxX(state) {
  const vPoints = state.x.map((_, i) =>
    scale(
      add(add(sub(state.z1[i], state.u1[i]), sub(state.z2[i], state.u2[i])), sub(state.z3[i], state.u3[i])),
      1 / 3,
    ),
  );
  const { assignments } = assignPoints(vPoints, state.cloud);
  const nextX = [];
  for (let i = 0; i < vPoints.length; i += 1) {
    const bucket = assignments[i];
    let sum = vec(0, 0);
    for (const point of bucket) {
      sum = add(sum, point);
    }
    const consensus = add(add(sub(state.z1[i], state.u1[i]), sub(state.z2[i], state.u2[i])), sub(state.z3[i], state.u3[i]));
    const denom = bucket.length + 3 * state.rho;
    nextX.push(scale(add(sum, scale(consensus, 1e-6)), 1 / Math.max(denom, 1e-6)));
  }
  return nextX;
}

function proxZ2(state) {
  const factor = state.rho / (state.rho + 2 * state.lambda);
  return state.x.map((point, i) => scale(add(point, state.u2[i]), factor));
}

function proxZ1(state, graph, weights) {
  const z = clonePoints(state.z1);
  const v = state.x.map((point, i) => add(point, state.u1[i]));
  const totalMass = Math.max(weights.reduce((sum, value) => sum + value, 0), 1);

  for (let step = 0; step < 7; step += 1) {
    const grad = z.map(() => vec(0, 0));

    for (let i = 0; i < z.length; i += 1) {
      const result = dijkstra(graph.adjacency, i);
      for (let j = i + 1; j < z.length; j += 1) {
        if (!Number.isFinite(result.dist[j])) {
          continue;
        }
        const coeff = 0.5 * (weights[i] * weights[j]) / (totalMass * totalMass);
        const path = reconstructPath(result.prev, i, j);
        for (let p = 0; p < path.length - 1; p += 1) {
          const a = path[p];
          const b = path[p + 1];
          const delta = sub(z[a], z[b]);
          const unit = normalize(delta);
          grad[a] = add(grad[a], scale(unit, 1e-6*coeff));
          grad[b] = add(grad[b], scale(unit, -1e-6*coeff));
        }
      }
    }

    for (let i = 0; i < z.length; i += 1) {
      const proxGrad = add(grad[i], scale(sub(z[i], v[i]), state.rho));
      z[i] = sub(z[i], scale(proxGrad, 0.12));
    }
  }

  return z;
}

function proxZ3(state, edges) {
  const z = clonePoints(state.z3);
  const v = state.x.map((point, i) => add(point, state.u3[i]));

  for (let step = 0; step < 7; step += 1) {
    const grad = z.map(() => vec(0, 0));
    for (const [a, b] of edges) {
      const delta = sub(z[a], z[b]);
      const unit = normalize(delta);
      grad[a] = add(grad[a], scale(unit, 1e-6));
      grad[b] = add(grad[b], scale(unit, -1e-6));
    }
    for (let i = 0; i < z.length; i += 1) {
      const proxGrad = add(grad[i], scale(sub(z[i], v[i]), state.rho));
      z[i] = sub(z[i], scale(proxGrad, 0.11));
    }
  }

  return z;
}

function residualSum(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += dist(a[i], b[i]);
  }
  return total;
}

function pointSetNormSum(points) {
  let total = 0;
  for (const point of points) {
    total += norm(point);
  }
  return total;
}

function computeResidualThresholds(state, absTolerance = 0.0005, relTolerance = 0.004) {
  const variableCount = state.x.length * 3;
  const xNorm = pointSetNormSum(state.x);
  const zNorm = pointSetNormSum(state.z1) + pointSetNormSum(state.z2) + pointSetNormSum(state.z3);
  const uNorm = pointSetNormSum(state.u1) + pointSetNormSum(state.u2) + pointSetNormSum(state.u3);
  return {
    primal: absTolerance * variableCount + relTolerance * Math.max(xNorm, zNorm),
    dual: absTolerance * variableCount + relTolerance * state.rho * uNorm,
  };
}

function computeMotionStats(current, previous) {
  let total = 0;
  let max = 0;
  for (let i = 0; i < current.length; i += 1) {
    const motion = dist(current[i], previous[i]);
    total += motion;
    if (motion > max) {
      max = motion;
    }
  }
  return {
    total,
    avg: current.length ? total / current.length : 0,
    max,
  };
}

function evaluateObjective(hubs, edges, cloud, params) {
  const assigned = assignPoints(hubs, cloud);
  const graph = graphFromEdges(hubs, edges);
  const fx = computeFx(hubs, assigned.assignments);
  const g1Info = computePathSmoothness(hubs, assigned.weights, graph);
  const g2 = computeG2(hubs, params.lambda);
  const g3 = computeGraphVariation(hubs, graph.edges, params.mu);
  const g4 = computeStationBuildCost(hubs, params.stationCost);

  console.log(g2, g3)
  return {
    objective: fx + g1Info.value + g2 + g3 + g4,
    fx,
    g1: g1Info.value,
    g2,
    g3,
    g4,
    assignments: assigned.assignments,
    weights: assigned.weights,
    highlightedPath: g1Info.highlighted.path,
    highlightedPair: g1Info.highlighted.pair,
    graph,
  };
}

function syncStateParams(state) {
  state.rho = Number(ui.rho.value);
  state.lambda = Number(ui.lambda.value);
  state.mu = Number(ui.mu.value);
  state.stationCost = Number(ui.stationCost.value);
}

function resetAdmmVariables(state, x, edges) {
  state.x = clonePoints(x);
  state.z1 = clonePoints(x);
  state.z2 = clonePoints(x);
  state.z3 = clonePoints(x);
  state.u1 = x.map(() => vec(0, 0));
  state.u2 = x.map(() => vec(0, 0));
  state.u3 = x.map(() => vec(0, 0));
  state.weights = Array(x.length).fill(0);
  state.edges = canonicalizeEdges(edges);
  state.highlightedPath = [0];
  state.highlightedPair = [0, 0];
}

function createBaseState(mode, cloud, x, edges) {
  const state = {
    mode,
    cloud,
    playing: true,
    iteration: 0,
    lambda: Number(ui.lambda.value),
    mu: Number(ui.mu.value),
    rho: Number(ui.rho.value),
    stationCost: Number(ui.stationCost.value),
    history: [],
    lastAction: "Initialised",
    forceStop: false,
  };
  resetAdmmVariables(state, x, edges);
  return state;
}

function createExplorerState() {
  const cloud = seededPointCloud(currentCloudSeed);
  const hubCount = Number(ui.hubCount.value);
  const x = sampleInitialHubs(cloud, hubCount);
  return createBaseState("explorer", cloud, x, []);
}

function createTuningState() {
  const cloud = seededPointCloud(currentCloudSeed);
  const startK = 3;
  const clustering = kMeans(cloud, startK, 16, 111);
  const graph = buildMaxGraph(clustering.centers);
  const state = createBaseState("tuning", cloud, clustering.centers, graph.edges);
  state.outerIteration = 0;
  state.lastAction = `Start with ${startK} hubs from KMeans`;
  return state;
}

function createCanvasState(cloud = customCloud, solveMode = canvasSolveMode) {
  const points = clonePoints(cloud);
  if (points.length === 0) {
    const state = createBaseState("canvas", [], [], []);
    state.canvasSolveMode = solveMode;
    state.outerIteration = 0;
    state.playing = false;
    state.lastAction = "Click inside the canvas to add points";
    return state;
  }

  let x = [];
  let edges = [];
  let lastAction = "";

  if (solveMode === "tuning") {
    const startK = Math.max(1, Math.min(3, points.length));
    const clustering = kMeans(points, startK, 16, 111);
    const graph = buildMaxGraph(clustering.centers);
    x = clustering.centers;
    edges = graph.edges;
    lastAction = `Start with ${x.length} hubs from KMeans on ${points.length} clicked points`;
  } else {
    const hubCount = Math.max(1, Math.min(Number(ui.hubCount.value), points.length));
    x = sampleInitialHubs(points, hubCount, 1337);
    lastAction = `Seeded ${x.length} hubs from ${points.length} clicked points`;
  }

  const state = createBaseState("canvas", points, x, edges);
  state.canvasSolveMode = solveMode;
  state.outerIteration = 0;
  state.lastAction = lastAction;
  return state;
}

function recordHistory(state, metrics, primal, dual) {
  state.weights = metrics.weights;
  state.highlightedPath = metrics.highlightedPath;
  state.highlightedPair = metrics.highlightedPair;
  state.history.push({
    objective: metrics.objective,
    primal,
    dual,
    fx: metrics.fx,
    g1: metrics.g1,
    g2: metrics.g2,
    g3: metrics.g3,
    g4: metrics.g4,
    maxWeight: Math.max(...metrics.weights, 0),
    hubCount: state.x.length,
    edgeCount: state.edges.length,
  });
  if (state.history.length > 180) {
    state.history.shift();
  }
}

function runAdmmIteration(state, graphMode = "knn", pushHistory = true, fixedEdges = null) {
  syncStateParams(state);
  const previousZ1 = clonePoints(state.z1);
  const previousZ2 = clonePoints(state.z2);
  const previousZ3 = clonePoints(state.z3);
  const assignmentBefore = assignPoints(state.x, state.cloud);
  state.weights = assignmentBefore.weights;

  state.x = proxX(state);

  const graph =
    fixedEdges !== null
      ? graphFromEdges(state.x, fixedEdges)
      : graphMode === "fixed"
        ? graphFromEdges(state.x, state.edges)
        : buildKnnGraph(state.x, 2);
  state.edges = graph.edges;
  state.z1 = proxZ1(state, graph, assignmentBefore.weights);
  state.z2 = proxZ2(state);
  state.z3 = proxZ3(state, graph.edges);

  for (let i = 0; i < state.x.length; i += 1) {
    state.u1[i] = add(state.u1[i], sub(state.x[i], state.z1[i]));
    state.u2[i] = add(state.u2[i], sub(state.x[i], state.z2[i]));
    state.u3[i] = add(state.u3[i], sub(state.x[i], state.z3[i]));
  }

  const metrics = evaluateObjective(state.x, state.edges, state.cloud, state);
  const primal = residualSum(state.x, state.z1) + residualSum(state.x, state.z2) + residualSum(state.x, state.z3);
  const dual =
    state.rho * residualSum(state.z1, previousZ1) +
    state.rho * residualSum(state.z2, previousZ2) +
    state.rho * residualSum(state.z3, previousZ3);

  state.iteration += 1;
  if (pushHistory) {
    recordHistory(state, metrics, primal, dual);
  }
  return { metrics, primal, dual };
}

function runFixedKConvergence(state, options = {}) {
  const {
    maxIterations = 48,
    minIterations = 6,
    primalAbsTolerance = 0.0005,
    primalRelTolerance = 0.004,
    pushHistory = true,
    graphMode = "knn",
    maxRuntimeMs = Infinity,
  } = options;

  let latest = null;
  let motion = { total: Infinity, avg: Infinity, max: Infinity };
  const fixedEdges = graphMode === "knn" ? buildKnnGraph(state.x, 2).edges : canonicalizeEdges(state.edges);
  state.edges = fixedEdges;
  const startTime = globalThis.performance?.now?.() ?? Date.now();

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const previousX = clonePoints(state.x);
    latest = runAdmmIteration(state, graphMode, pushHistory, fixedEdges);
    motion = computeMotionStats(state.x, previousX);
    const thresholds = computeResidualThresholds(state, primalAbsTolerance, primalRelTolerance);
    const residualSettled = latest.primal <= thresholds.primal && latest.dual <= thresholds.dual;
    if (iteration + 1 >= minIterations && residualSettled) {
      return { ...latest, motion, settled: true, iterations: iteration + 1 };
    }
    if ((globalThis.performance?.now?.() ?? Date.now()) - startTime >= maxRuntimeMs) {
      return { ...latest, motion, settled: false, iterations: iteration + 1, timedOut: true };
    }
  }

  return { ...latest, motion, settled: false, iterations: maxIterations };
}

function optimizeEdgesForState(state, metrics = null) {
  if (state.x.length <= 1) {
    state.edges = [];
    const finalMetrics = metrics ?? evaluateObjective(state.x, state.edges, state.cloud, state);
    return { edges: [], metrics: finalMetrics, removedEdges: 0, improved: false };
  }

  let currentEdges = canonicalizeEdges(state.edges);
  let canReuseMetrics = Boolean(metrics) && currentEdges.length === state.edges.length;
  if (!isGraphConnected(state.x, currentEdges)) {
    currentEdges = buildCompleteGraph(state.x).edges;
    canReuseMetrics = false;
  }

  let currentMetrics = canReuseMetrics ? metrics : evaluateObjective(state.x, currentEdges, state.cloud, state);
  let improved = false;
  let removedEdges = 0;
  const startTime = globalThis.performance?.now?.() ?? Date.now();
  const maxRuntimeMs = 24;

  while (currentEdges.length > state.x.length - 1) {
    if ((globalThis.performance?.now?.() ?? Date.now()) - startTime >= maxRuntimeMs) {
      break;
    }
    let bestCandidate = null;

    for (let edgeIndex = 0; edgeIndex < currentEdges.length; edgeIndex += 1) {
      if ((globalThis.performance?.now?.() ?? Date.now()) - startTime >= maxRuntimeMs) {
        break;
      }
      const candidateEdges = currentEdges.filter((_, index) => index !== edgeIndex);
      if (!isGraphConnected(state.x, candidateEdges)) {
        continue;
      }

      const candidateMetrics = evaluateObjective(state.x, candidateEdges, state.cloud, state);
      const delta = currentMetrics.objective - candidateMetrics.objective;
      if (delta > 1e-6 && (!bestCandidate || delta > bestCandidate.delta)) {
        bestCandidate = {
          edges: candidateEdges,
          metrics: candidateMetrics,
          delta,
        };
      }
    }

    if (!bestCandidate) {
      break;
    }

    currentEdges = bestCandidate.edges;
    currentMetrics = bestCandidate.metrics;
    improved = true;
    removedEdges += 1;
  }

  state.edges = currentEdges;
  return { edges: currentEdges, metrics: currentMetrics, removedEdges, improved };
}

function runConvergedAdmmWithEdgeOptimization(state, options = {}) {
  const convergenceOptions = {
    maxIterations: 48,
    minIterations: 6,
    primalAbsTolerance: 0.0005,
    primalRelTolerance: 0.004,
    pushHistory: true,
    graphMode: "knn",
    maxRuntimeMs: Infinity,
    ...options,
  };

  const convergence = runFixedKConvergence(state, convergenceOptions);
  const edgeOptimization = optimizeEdgesForState(state, convergence.metrics);

  if (!edgeOptimization.improved) {
    return { ...convergence, edgeOptimization };
  }

  resetAdmmVariables(state, state.x, edgeOptimization.edges);
  const settled = runFixedKConvergence(state, {
    ...convergenceOptions,
    graphMode: "fixed",
  });
  return { ...settled, edgeOptimization };
}

function runFinalAdmmCompletion(state) {
  return runFixedKConvergence(state, {
    maxIterations: 96,
    minIterations: 12,
    primalAbsTolerance: 0.00025,
    primalRelTolerance: 0.002,
    graphMode: "knn",
    maxRuntimeMs: 40,
  });
}

function simulateFixedHubRefinement(baseState, x, edges) {
  const state = createBaseState("simulation", baseState.cloud, x, edges);
  state.lambda = baseState.lambda;
  state.mu = baseState.mu;
  state.rho = baseState.rho;
  state.stationCost = baseState.stationCost;
  const convergence = runConvergedAdmmWithEdgeOptimization(state, { pushHistory: false, graphMode: "knn" });
  return { state, metrics: convergence.metrics, convergence };
}

function simulateFinalAdmmCompletion(baseState, x, edges) {
  const state = createBaseState("simulation", baseState.cloud, x, edges);
  state.lambda = baseState.lambda;
  state.mu = baseState.mu;
  state.rho = baseState.rho;
  state.stationCost = baseState.stationCost;
  const completion = runFinalAdmmCompletion(state);
  return { state, metrics: completion.metrics, completion };
}

function buildSplitProposal(state) {
  const currentEval = evaluateObjective(state.x, state.edges, state.cloud, state);
  const clusterEnergies = computeClusterEnergies(currentEval.assignments, state.x);
  let clusterIndex = -1;
  let bestEnergy = -Infinity;
  for (let i = 0; i < clusterEnergies.length; i += 1) {
    if (currentEval.assignments[i].length < 2) {
      continue;
    }
    if (clusterEnergies[i] > bestEnergy) {
      bestEnergy = clusterEnergies[i];
      clusterIndex = i;
    }
  }
  if (clusterIndex === -1) {
    return null;
  }

  const clusterPoints = currentEval.assignments[clusterIndex];
  const splitSeed = state.outerIteration + clusterIndex * 19 + state.x.length * 7;
  const split = kMeans(clusterPoints, 2, 12, splitSeed);
  if (split.centers.length < 2) {
    return null;
  }

  let best = null;
  for (const candidate of split.centers) {
    const proposalHubs = clonePoints(state.x);
    proposalHubs.push(candidate);
    const candidateIndex = proposalHubs.length - 1;

    for (let anchorIndex = 0; anchorIndex < state.x.length; anchorIndex += 1) {
      const proposalEdges = canonicalizeEdges(state.edges.concat([[anchorIndex, candidateIndex]]));
      const simulation = simulateFixedHubRefinement(state, proposalHubs, proposalEdges);
      const delta = currentEval.objective - simulation.metrics.objective;

      if (!best || delta > best.delta) {
        best = {
          delta,
          clusterIndex,
          clusterEnergy: bestEnergy,
          candidate,
          anchorIndex,
          simulation,
        };
      }
    }
  }

  return best;
}

function adoptSimulationState(target, simulation, lastAction) {
  resetAdmmVariables(target, simulation.state.x, simulation.state.edges);
  target.lambda = simulation.state.lambda;
  target.mu = simulation.state.mu;
  target.rho = simulation.state.rho;
  target.stationCost = simulation.state.stationCost;
  target.lastAction = lastAction;
}

function stepExplorerState(state) {
  if (state.x.length === 0 || state.cloud.length === 0) {
    state.playing = false;
    return;
  }
  runAdmmIteration(state, "knn", true);
}

function stepTuningState(state) {
  if (state.x.length === 0 || state.cloud.length === 0) {
    state.playing = false;
    return;
  }
  syncStateParams(state);
  let convergence = runConvergedAdmmWithEdgeOptimization(state, { graphMode: "knn" });
  const baselineX = clonePoints(state.x);
  const baselineEdges = canonicalizeEdges(state.edges);
  const baselineIteration = state.iteration;
  const baselineWeights = state.weights.slice();
  const baselineHighlightedPath = [...state.highlightedPath];
  const baselineHighlightedPair = [...state.highlightedPair];
  const baselineHistoryLength = state.history.length;
  const baselineFinal = simulateFinalAdmmCompletion(state, baselineX, baselineEdges);
  const baselineFinalObjective = baselineFinal.metrics.objective;

  state.outerIteration += 1;

  const splitProposal = buildSplitProposal(state);
  if (splitProposal && splitProposal.delta > 0) {
    adoptSimulationState(
      state,
      splitProposal.simulation,
      `Added hub from Voronoi split ${splitProposal.clusterIndex} via ${splitProposal.anchorIndex} (ΔF=${splitProposal.delta.toFixed(3)})`,
    );
    convergence = runConvergedAdmmWithEdgeOptimization(state, { graphMode: "knn" });
    const finalConvergence = runFinalAdmmCompletion(state);
    if (finalConvergence.metrics.objective + 1e-6 < baselineFinalObjective || state.forceStop) {
      const finalDelta = baselineFinalObjective - finalConvergence.metrics.objective;
      const edgeNote = convergence.edgeOptimization.improved
        ? `; pruned ${convergence.edgeOptimization.removedEdges} edges after convergence`
        : "";
      const completionLabel = finalConvergence.settled
        ? `final ADMM completed in ${finalConvergence.iterations} sweeps`
        : finalConvergence.timedOut
          ? `final ADMM stopped at the runtime budget after ${finalConvergence.iterations} sweeps`
          : `final ADMM hit the sweep cap after ${finalConvergence.iterations} sweeps`;
      state.lastAction = `${state.lastAction}; settled in ${convergence.iterations} ADMM sweeps; ${completionLabel}; final ΔF=${finalDelta.toFixed(3)}${edgeNote}`;
    } else {
      resetAdmmVariables(state, baselineX, baselineEdges);
      state.iteration = baselineIteration;
      state.weights = baselineWeights;
      state.highlightedPath = baselineHighlightedPath;
      state.highlightedPair = baselineHighlightedPair;
      state.history.length = baselineHistoryLength;
      state.lastAction = `Rejected hub split at outer step ${state.outerIteration}; final ADMM objective rose from ${baselineFinalObjective.toFixed(3)} to ${finalConvergence.metrics.objective.toFixed(3)}`;
      state.forceStop = true
    }
  } else {
    const settleLabel = convergence.settled ? "settled" : "hit the ADMM sweep cap";
    const finalConvergence = runFinalAdmmCompletion(state);
    const completionLabel = finalConvergence.settled
      ? `final ADMM completed in ${finalConvergence.iterations} sweeps`
      : finalConvergence.timedOut
        ? `final ADMM stopped at the runtime budget after ${finalConvergence.iterations} sweeps`
        : `final ADMM hit the sweep cap after ${finalConvergence.iterations} sweeps`;
    state.lastAction = `No improving hub split after ${settleLabel} at outer step ${state.outerIteration}; ${completionLabel}`;
    state.playing = false;
  }
}

function stepExplorer() {
  stepExplorerState(explorerState);
}

function stepTuning() {
  stepTuningState(tuningState);
}

function stepCanvas() {
  if (canvasSolveMode === "tuning") {
    stepTuningState(canvasState);
  } else {
    stepExplorerState(canvasState);
  }
}

const camera = {
  zoom: 1,
  offsetX: 0,
  offsetY: 0,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function projectPoint(point, bounds, size, view = camera) {
  const pad = 46;
  const sx = (size.width - pad * 2) / (bounds.maxX - bounds.minX);
  const sy = (size.height - pad * 2) / (bounds.maxY - bounds.minY);
  const scaleValue = Math.min(sx, sy);
  const baseX = pad + (point.x - bounds.minX) * scaleValue;
  const baseY = size.height - pad - (point.y - bounds.minY) * scaleValue;
  const centerX = size.width * 0.5;
  const centerY = size.height * 0.5;
  return {
    x: centerX + (baseX - centerX) * view.zoom + view.offsetX,
    y: centerY + (baseY - centerY) * view.zoom + view.offsetY,
  };
}

function unprojectPoint(screenPoint, bounds, size, view = camera) {
  const pad = 46;
  const sx = (size.width - pad * 2) / (bounds.maxX - bounds.minX);
  const sy = (size.height - pad * 2) / (bounds.maxY - bounds.minY);
  const scaleValue = Math.min(sx, sy);
  const centerX = size.width * 0.5;
  const centerY = size.height * 0.5;
  const baseX = centerX + (screenPoint.x - view.offsetX - centerX) / view.zoom;
  const baseY = centerY + (screenPoint.y - view.offsetY - centerY) / view.zoom;
  return vec(
    bounds.minX + (baseX - pad) / scaleValue,
    bounds.minY + (size.height - pad - baseY) / scaleValue,
  );
}

function computeBounds(cloud, hubs, fallback = null) {
  const all = cloud.concat(hubs);
  if (all.length === 0) {
    return fallback ?? { minX: -2.4, maxX: 2.4, minY: -2.1, maxY: 2.1 };
  }
  const xs = all.map((point) => point.x);
  const ys = all.map((point) => point.y);
  return {
    minX: Math.min(...xs) - 0.3,
    maxX: Math.max(...xs) + 0.3,
    minY: Math.min(...ys) - 0.3,
    maxY: Math.max(...ys) + 0.3,
  };
}

function getSceneBounds(state) {
  const canvasFallback = { minX: -2.4, maxX: 2.4, minY: -2.1, maxY: 2.1 };
  if (state.mode === "canvas") {
    const fitted = computeBounds(state.cloud, state.x, canvasFallback);
    return {
      minX: Math.min(fitted.minX, canvasFallback.minX),
      maxX: Math.max(fitted.maxX, canvasFallback.maxX),
      minY: Math.min(fitted.minY, canvasFallback.minY),
      maxY: Math.max(fitted.maxY, canvasFallback.maxY),
    };
  }
  return computeBounds(state.cloud, state.x, canvasFallback);
}

function scaleVisualSize(base, zoom, min, max, exponent = 0.55) {
  return clamp(base * zoom ** exponent, min, max);
}

function drawScene(state) {
  const { width, height } = sceneCanvas;
  sceneCtx.clearRect(0, 0, width, height);

  const bounds = getSceneBounds(state);
  const { assignments } = state.x.length > 0 ? assignPoints(state.x, state.cloud) : { assignments: [] };
  const clusterEnergies = state.x.length > 0 ? computeClusterEnergies(assignments, state.x) : [];
  const voronoiCells = showVoronoiOverlay ? computeVoronoiCells(state.x, bounds) : [];

  sceneCtx.fillStyle = "#fff8ef";
  sceneCtx.fillRect(0, 0, width, height);

  sceneCtx.strokeStyle = "rgba(20, 16, 13, 0.08)";
  sceneCtx.lineWidth = scaleVisualSize(1, camera.zoom, 0.75, 2.2, 0.2);
  for (let i = 0; i < 9; i += 1) {
    const t = i / 8;
    const y = 40 + t * (height - 80);
    const x = 40 + t * (width - 80);
    sceneCtx.beginPath();
    sceneCtx.moveTo(34, y);
    sceneCtx.lineTo(width - 34, y);
    sceneCtx.stroke();
    sceneCtx.beginPath();
    sceneCtx.moveTo(x, 34);
    sceneCtx.lineTo(x, height - 34);
    sceneCtx.stroke();
  }

  if (state.mode === "canvas" && state.cloud.length === 0) {
    sceneCtx.fillStyle = "rgba(29, 26, 22, 0.62)";
    sceneCtx.font = '18px "IBM Plex Mono"';
    sceneCtx.textAlign = "center";
    sceneCtx.fillText("Click to place point-cloud samples", width * 0.5, height * 0.48);
    sceneCtx.fillText("Press Clear Canvas to start over", width * 0.5, height * 0.53);
    return;
  }

  if (showVoronoiOverlay) {
    for (let hubIndex = 0; hubIndex < voronoiCells.length; hubIndex += 1) {
      const cell = voronoiCells[hubIndex];
      if (cell.length < 3) {
        continue;
      }
      sceneCtx.beginPath();
      const start = projectPoint(cell[0], bounds, { width, height });
      sceneCtx.moveTo(start.x, start.y);
      for (let pointIndex = 1; pointIndex < cell.length; pointIndex += 1) {
        const point = projectPoint(cell[pointIndex], bounds, { width, height });
        sceneCtx.lineTo(point.x, point.y);
      }
      sceneCtx.closePath();
      sceneCtx.fillStyle = `hsla(${(hubIndex * 53) % 360} 72% 54% / 0.08)`;
      sceneCtx.strokeStyle = `hsla(${(hubIndex * 53) % 360} 58% 34% / 0.42)`;
      sceneCtx.lineWidth = scaleVisualSize(1.2, camera.zoom, 0.9, 2.8, 0.25);
      sceneCtx.fill();
      sceneCtx.stroke();

      const centroid = polygonCentroid(cell);
      if (centroid) {
        const labelPoint = projectPoint(centroid, bounds, { width, height });
        sceneCtx.fillStyle = "rgba(16, 16, 15, 0.88)";
        sceneCtx.font = `${Math.round(scaleVisualSize(11, camera.zoom, 9, 16, 0.3))}px IBM Plex Mono`;
        sceneCtx.textAlign = "center";
        sceneCtx.fillText(`E${hubIndex}: ${(clusterEnergies[hubIndex] ?? 0).toFixed(2)}`, labelPoint.x, labelPoint.y);
      }
    }
  }

  for (let hubIndex = 0; hubIndex < assignments.length; hubIndex += 1) {
    const color = `hsla(${(hubIndex * 53) % 360} 70% 52% / 0.12)`;
    sceneCtx.fillStyle = color;
    for (const point of assignments[hubIndex]) {
      const projected = projectPoint(point, bounds, { width, height });
      sceneCtx.beginPath();
      sceneCtx.arc(projected.x, projected.y, scaleVisualSize(100, camera.zoom, 1.5, 4.4, 100), 0, TAU);
      sceneCtx.fill();
    }
  }

  sceneCtx.strokeStyle = "rgba(35, 117, 111, 0.42)";
  sceneCtx.lineWidth = scaleVisualSize(1.3, camera.zoom, 1, 3.4, 0.35);
  for (const [a, b] of state.edges) {
    const pa = projectPoint(state.x[a], bounds, { width, height });
    const pb = projectPoint(state.x[b], bounds, { width, height });
    sceneCtx.beginPath();
    sceneCtx.moveTo(pa.x, pa.y);
    sceneCtx.lineTo(pb.x, pb.y);
    sceneCtx.stroke();
  }

  if (state.highlightedPath.length > 1) {
    sceneCtx.strokeStyle = "#d77b2a";
    sceneCtx.lineWidth = scaleVisualSize(4, camera.zoom, 2.4, 7.2, 0.4);
    sceneCtx.lineJoin = "round";
    sceneCtx.beginPath();
    const start = projectPoint(state.x[state.highlightedPath[0]], bounds, { width, height });
    sceneCtx.moveTo(start.x, start.y);
    for (let i = 1; i < state.highlightedPath.length; i += 1) {
      const point = projectPoint(state.x[state.highlightedPath[i]], bounds, { width, height });
      sceneCtx.lineTo(point.x, point.y);
    }
    sceneCtx.stroke();
  }

  for (let i = 0; i < state.x.length; i += 1) {
    const projected = projectPoint(state.x[i], bounds, { width, height });
    const weight = state.weights[i] ?? 0;
    const hubRadius = scaleVisualSize(6.5 + Math.min(weight / 24, 8), camera.zoom, 4.5, 19, 0.55);
    sceneCtx.fillStyle = "#10100f";
    sceneCtx.beginPath();
    sceneCtx.arc(projected.x, projected.y, hubRadius, 0, TAU);
    sceneCtx.fill();
    sceneCtx.fillStyle = "#fff3e0";
    sceneCtx.font = `${Math.round(scaleVisualSize(12, camera.zoom, 9, 18, 0.35))}px IBM Plex Mono`;
    sceneCtx.textAlign = "center";
    sceneCtx.fillText(String(i), projected.x, projected.y + scaleVisualSize(4, camera.zoom, 3, 6, 0.3));
  }
}

function drawChart(ctx, canvas, seriesList, colors, labels) {
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fff7ed";
  ctx.fillRect(0, 0, width, height);

  const pad = 26;
  const values = seriesList.flat();
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const span = Math.max(max - min, 1e-6);

  ctx.strokeStyle = "rgba(24, 22, 18, 0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = pad + (i / 4) * (height - pad * 2);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  seriesList.forEach((series, index) => {
    ctx.strokeStyle = colors[index];
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    series.forEach((value, i) => {
      const x = pad + (i / Math.max(series.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });

  labels.forEach((label, index) => {
    ctx.fillStyle = colors[index];
    ctx.font = "12px IBM Plex Mono";
    ctx.fillText(label, 18 + index * 112, 18);
  });
}

let activeMode = "explorer";
let currentCloudSeed = 12;
let customCloud = [];
let canvasSolveMode = "explorer";
let showVoronoiOverlay = false;
let explorerState = createExplorerState();
let tuningState = createTuningState();
let canvasState = createCanvasState();

function getActiveState() {
  if (activeMode === "explorer") {
    return explorerState;
  }
  if (activeMode === "tuning") {
    return tuningState;
  }
  return canvasState;
}

function updateUi() {
  const state = getActiveState();
  const current = state.history[state.history.length - 1] || {
    objective: 0,
    primal: 0,
    dual: 0,
    fx: 0,
    g1: 0,
    g2: 0,
    g3: 0,
    g4: 0,
    maxWeight: 0,
    hubCount: state.x.length,
    edgeCount: state.edges.length,
  };

  const showOuterIteration = state.mode === "tuning" || (state.mode === "canvas" && canvasSolveMode === "tuning");
  ui.iterValue.textContent = String(showOuterIteration ? state.outerIteration : state.iteration);
  ui.objectiveValue.textContent = current.objective.toFixed(3);
  ui.primalValue.textContent = current.primal.toFixed(3);
  ui.dualValue.textContent = current.dual.toFixed(3);
  ui.fxValue.textContent = current.fx.toFixed(3);
  ui.g1Value.textContent = current.g1.toFixed(3);
  ui.g2Value.textContent = current.g2.toFixed(3);
  ui.g3Value.textContent = current.g3.toFixed(3);
  ui.g4Value.textContent = current.g4.toFixed(3);
  ui.playPause.textContent = state.playing ? "Pause" : "Play";
  syncSliderLabels();
  ui.modeSummary.innerHTML = modeCopy[activeMode];
  ui.explorerTab.classList.toggle("is-active", activeMode === "explorer");
  ui.tuningTab.classList.toggle("is-active", activeMode === "tuning");
  ui.canvasTab.classList.toggle("is-active", activeMode === "canvas");
  ui.canvasSolverGroup.classList.toggle("is-hidden", activeMode !== "canvas");
  ui.canvasExplorerMode.classList.toggle("is-active", canvasSolveMode === "explorer");
  ui.canvasTuningMode.classList.toggle("is-active", canvasSolveMode === "tuning");
  ui.hubCountGroup.classList.toggle(
    "is-hidden",
    activeMode === "tuning" || (activeMode === "canvas" && canvasSolveMode === "tuning"),
  );
  ui.randomizeButton.textContent = activeMode === "canvas" ? "Clear Canvas" : "Randomise";
  ui.voronoiToggle.textContent = showVoronoiOverlay ? "Hide Voronoi" : "Show Voronoi";
  ui.voronoiToggle.classList.toggle("is-active", showVoronoiOverlay);
  sceneCanvas.classList.toggle("is-editable", activeMode === "canvas");

  if (state.mode === "tuning") {
    ui.weightLabel.textContent = "Hub / edge count";
    ui.weightValue.textContent = `${current.hubCount} / ${current.edgeCount}`;
    ui.pathLabel.textContent = "Last tuning action";
    ui.pathValue.textContent = state.lastAction;
  } else if (state.mode === "canvas") {
    ui.weightLabel.textContent = "Point / hub count";
    ui.weightValue.textContent = `${state.cloud.length} / ${state.x.length}`;
    ui.pathLabel.textContent = canvasSolveMode === "tuning" ? "Canvas tuning action" : "Canvas action";
    ui.pathValue.textContent = state.lastAction;
  } else {
    ui.weightLabel.textContent = "Largest hub weight";
    ui.weightValue.textContent = String(current.maxWeight);
    ui.pathLabel.textContent = "Highlighted path";
    ui.pathValue.textContent = `${state.highlightedPair[0]} → ${state.highlightedPair[1]}`;
  }

  drawScene(state);
  drawChart(
    objectiveCtx,
    objectiveCanvas,
    [state.history.map((row) => row.objective), state.history.map((row) => row.fx)],
    ["#cb5e38", "#23756f"],
    ["objective", "f(x)"],
  );
  drawChart(
    residualCtx,
    residualCanvas,
    [state.history.map((row) => row.primal), state.history.map((row) => row.dual)],
    ["#d77b2a", "#3f5d53"],
    ["primal", "dual"],
  );
}

function resetState(mode = activeMode) {
  if (mode === "explorer") {
    explorerState = createExplorerState();
    for (let i = 0; i < 3; i += 1) {
      stepExplorer();
    }
  } else if (mode === "tuning") {
    tuningState = createTuningState();
    for (let i = 0; i < 2; i += 1) {
      stepTuning();
    }
  } else {
    canvasState = createCanvasState();
    if (canvasState.x.length > 0) {
      const warmupSteps = canvasSolveMode === "tuning" ? 2 : 3;
      for (let i = 0; i < warmupSteps; i += 1) {
        stepCanvas();
      }
    }
  }
  updateUi();
}

function randomizeCloud(mode = activeMode) {
  if (mode === "canvas") {
    customCloud = [];
    canvasState = createCanvasState();
    updateUi();
    return;
  }
  currentCloudSeed = randomSeed();
  explorerState = createExplorerState();
  tuningState = createTuningState();
  resetState(mode);
}

function switchMode(mode) {
  activeMode = mode;
  if (getActiveState().history.length === 0) {
    resetState(mode);
    return;
  }
  updateUi();
}

function nudgeCamera(dx = 0, dy = 0) {
  camera.offsetX += dx;
  camera.offsetY += dy;
  updateUi();
}

function changeZoom(factor) {
  camera.zoom = clamp(camera.zoom * factor, 0.65, 3.5);
  updateUi();
}

const sliderConfigs = [
  { input: ui.hubCount, value: ui.hubCountValue, format: (raw) => String(Math.round(Number(raw))) },
  { input: ui.rho, value: ui.rhoValue, format: (raw) => Number(raw).toFixed(2) },
  { input: ui.lambda, value: ui.lambdaValue, format: (raw) => Number(raw).toFixed(2) },
  { input: ui.mu, value: ui.muValue, format: (raw) => Number(raw).toFixed(2) },
  { input: ui.stationCost, value: ui.stationCostValue, format: (raw) => Number(raw).toFixed(2) },
];

function syncSliderLabels() {
  for (const { input, value, format } of sliderConfigs) {
    value.textContent = format(input.value);
  }
}

function quantizeSliderValue(slider, rawValue) {
  const min = Number(slider.min);
  const max = Number(slider.max);
  const step = slider.step === "any" || slider.step === "" ? null : Number(slider.step);
  const clamped = clamp(rawValue, min, max);

  if (!step || step <= 0) {
    return clamped;
  }

  const base = Number.isFinite(min) ? min : 0;
  const snapped = Math.round((clamped - base) / step) * step + base;
  const decimals = (slider.step.split(".")[1] || "").length;
  return Number(snapped.toFixed(decimals));
}

function setSliderValue(slider, rawValue) {
  const nextValue = quantizeSliderValue(slider, rawValue);
  slider.value = String(nextValue);
  slider.dispatchEvent(new Event("input", { bubbles: true }));
  slider.dispatchEvent(new Event("change", { bubbles: true }));
}

function promptForSliderValue(slider) {
  const nextRaw = globalThis.prompt?.(
    `Enter a value for ${slider.id} (${slider.min} to ${slider.max}${slider.step ? `, step ${slider.step}` : ""})`,
    slider.value,
  );
  if (nextRaw === null) {
    return;
  }
  const parsed = Number(nextRaw.trim());
  if (!Number.isFinite(parsed)) {
    return;
  }
  setSliderValue(slider, parsed);
}

ui.playPause.addEventListener("click", () => {
  const state = getActiveState();
  state.playing = !state.playing;
  updateUi();
});

ui.stepButton.addEventListener("click", () => {
  const state = getActiveState();
  state.playing = false;
  if (activeMode === "explorer") {
    stepExplorer();
  } else if (activeMode === "tuning") {
    stepTuning();
  } else {
    stepCanvas();
  }
  updateUi();
});

ui.resetButton.addEventListener("click", () => resetState(activeMode));
ui.randomizeButton.addEventListener("click", () => randomizeCloud(activeMode));
ui.voronoiToggle.addEventListener("click", () => {
  showVoronoiOverlay = !showVoronoiOverlay;
  updateUi();
});
ui.explorerTab.addEventListener("click", () => switchMode("explorer"));
ui.tuningTab.addEventListener("click", () => switchMode("tuning"));
ui.canvasTab.addEventListener("click", () => switchMode("canvas"));
ui.canvasExplorerMode.addEventListener("click", () => {
  canvasSolveMode = "explorer";
  canvasState = createCanvasState();
  resetState(activeMode);
});
ui.canvasTuningMode.addEventListener("click", () => {
  canvasSolveMode = "tuning";
  canvasState = createCanvasState();
  resetState(activeMode);
});
ui.zoomIn.addEventListener("click", () => changeZoom(1.2));
ui.zoomOut.addEventListener("click", () => changeZoom(1 / 1.2));
ui.panUp.addEventListener("click", () => nudgeCamera(0, 34));
ui.panDown.addEventListener("click", () => nudgeCamera(0, -34));
ui.panLeft.addEventListener("click", () => nudgeCamera(34, 0));
ui.panRight.addEventListener("click", () => nudgeCamera(-34, 0));
sceneCanvas.addEventListener("click", (event) => {
  if (activeMode !== "canvas") {
    return;
  }
  const rect = sceneCanvas.getBoundingClientRect();
  const state = getActiveState();
  const bounds = getSceneBounds(state);
  const screenPoint = {
    x: ((event.clientX - rect.left) / rect.width) * sceneCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * sceneCanvas.height,
  };
  customCloud.push(unprojectPoint(screenPoint, bounds, { width: sceneCanvas.width, height: sceneCanvas.height }));
  canvasState = createCanvasState();
  canvasState.playing = true;
  canvasState.lastAction =
    canvasSolveMode === "tuning"
      ? `Added point ${customCloud.length}; hub tuning restarted with ${canvasState.x.length} hubs`
      : `Added point ${customCloud.length}; solving with ${canvasState.x.length} hubs`;
  updateUi();
});

for (const slider of [ui.hubCount, ui.rho, ui.lambda, ui.mu, ui.stationCost]) {
  slider.addEventListener("input", () => {
    syncSliderLabels();
  });
}

for (const { input, value } of sliderConfigs) {
  value.addEventListener("click", () => promptForSliderValue(input));
  value.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    promptForSliderValue(input);
  });
}

ui.hubCount.addEventListener("change", () => {
  explorerState = createExplorerState();
  tuningState = createTuningState();
  canvasState = createCanvasState();
  resetState(activeMode);
});

let lastTick = 0;
function animate(timestamp) {
  if (timestamp - lastTick > 150) {
    const state = getActiveState();
    if (state.playing) {
      if (activeMode === "explorer") {
        stepExplorer();
      } else if (activeMode === "tuning") {
        stepTuning();
      } else {
        stepCanvas();
      }
    }
    updateUi();
    lastTick = timestamp;
  }
  requestAnimationFrame(animate);
}

resetState("explorer");
requestAnimationFrame(animate);
