import { buildKnnGraph, buildCompleteGraph, canonicalizeEdges, dijkstra, graphFromEdges, isGraphConnected, reconstructPath } from "../core/graph.js";
import { assignPoints, computeClusterEnergies, evaluateObjective } from "../core/objective.js";
import { add, clonePoints, dist, scale, sub, vec } from "../core/math.js";

function dotPoints(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += a[i].x * b[i].x + a[i].y * b[i].y;
  }
  return total;
}

function pointVectorNorm(points) {
  return Math.sqrt(Math.max(dotPoints(points, points), 0));
}

function solveConjugateGradient(applyOperator, rhs, tolerance = 1e-6, maxIterations = 128) {
  let x = rhs.map(() => vec(0, 0));
  let residual = clonePoints(rhs);
  let direction = clonePoints(residual);
  let residualNormSq = dotPoints(residual, residual);

  if (residualNormSq <= tolerance * tolerance) {
    return x;
  }

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const applied = applyOperator(direction);
    const denom = Math.max(dotPoints(direction, applied), 1e-12);
    const alpha = residualNormSq / denom;
    x = x.map((point, index) => add(point, scale(direction[index], alpha)));
    residual = residual.map((point, index) => sub(point, scale(applied[index], alpha)));
    const nextResidualNormSq = dotPoints(residual, residual);
    if (nextResidualNormSq <= tolerance * tolerance) {
      return x;
    }
    const beta = nextResidualNormSq / Math.max(residualNormSq, 1e-12);
    direction = residual.map((point, index) => add(point, scale(direction[index], beta)));
    residualNormSq = nextResidualNormSq;
  }

  return x;
}

function applyWeightedLaplacian(points, edgeWeights) {
  const result = points.map(() => vec(0, 0));
  for (const { a, b, weight } of edgeWeights) {
    const delta = sub(points[a], points[b]);
    result[a] = add(result[a], scale(delta, weight));
    result[b] = sub(result[b], scale(delta, weight));
  }
  return result;
}

function solveQuadraticProx(v, rho, edgeWeights) {
  if (v.length === 0) {
    return [];
  }
  if (edgeWeights.length === 0) {
    return clonePoints(v);
  }

  const rhs = v.map((point) => scale(point, rho));
  const applyOperator = (points) => {
    const laplacian = applyWeightedLaplacian(points, edgeWeights);
    return points.map((point, index) => add(scale(point, rho), scale(laplacian[index], 2)));
  };

  return solveConjugateGradient(applyOperator, rhs, 1e-6, Math.max(64, v.length * 8));
}

function buildPathEdgeWeights(graph, weights) {
  const totalMass = Math.max(weights.reduce((sum, value) => sum + value, 0), 1);
  const edgeWeights = new Map();

  for (let i = 0; i < weights.length; i += 1) {
    const result = dijkstra(graph.adjacency, i);
    for (let j = i + 1; j < weights.length; j += 1) {
      if (!Number.isFinite(result.dist[j])) {
        continue;
      }
      const coeff = 0.5 * (weights[i] * weights[j]) / (totalMass * totalMass);
      if (coeff <= 0) {
        continue;
      }
      const path = reconstructPath(result.prev, i, j);
      for (let p = 0; p < path.length - 1; p += 1) {
        const a = Math.min(path[p], path[p + 1]);
        const b = Math.max(path[p], path[p + 1]);
        const key = `${a}-${b}`;
        edgeWeights.set(key, (edgeWeights.get(key) ?? 0) + coeff);
      }
    }
  }

  return [...edgeWeights.entries()].map(([key, weight]) => {
    const [a, b] = key.split("-").map(Number);
    return { a, b, weight };
  });
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
    nextX.push(scale(add(sum, scale(consensus, state.rho)), 1 / Math.max(denom, 1e-6)));
  }
  return nextX;
}

function proxZ2(state) {
  const factor = state.rho / (state.rho + 2 * state.lambda);
  return state.x.map((point, i) => scale(add(point, state.u2[i]), factor));
}

function proxZ1(state, graph, weights) {
  const v = state.x.map((point, i) => add(point, state.u1[i]));
  const edgeWeights = buildPathEdgeWeights(graph, weights);
  return solveQuadraticProx(v, state.rho, edgeWeights);
}

function proxZ3(state, edges) {
  const v = state.x.map((point, i) => add(point, state.u3[i]));
  const edgeWeights = edges.map(([a, b]) => ({ a, b, weight: state.mu }));
  return solveQuadraticProx(v, state.rho, edgeWeights);
}

function residualSum(a, b) {
  let totalSq = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = sub(a[i], b[i]);
    totalSq += delta.x * delta.x + delta.y * delta.y;
  }
  return Math.sqrt(totalSq);
}

function computeResidualThresholds(state, absTolerance = 0.0005, relTolerance = 0.004) {
  const variableCount = state.x.length * 3;
  const xNorm = pointVectorNorm(state.x);
  const zNorm = Math.sqrt(
    pointVectorNorm(state.z1) ** 2 + pointVectorNorm(state.z2) ** 2 + pointVectorNorm(state.z3) ** 2,
  );
  const uNorm = Math.sqrt(
    pointVectorNorm(state.u1) ** 2 + pointVectorNorm(state.u2) ** 2 + pointVectorNorm(state.u3) ** 2,
  );
  return {
    primal: Math.sqrt(variableCount) * absTolerance + relTolerance * Math.max(Math.sqrt(3) * xNorm, zNorm),
    dual: Math.sqrt(variableCount) * absTolerance + relTolerance * state.rho * uNorm,
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

function sampleWithoutReplacement(points, count, rng) {
  if (count >= points.length) {
    return points.slice();
  }
  const indices = new Set();
  while (indices.size < count) {
    indices.add(Math.floor(rng() * points.length));
  }
  return [...indices].map((index) => points[index]);
}

function computeSgdGradient(x, cloud, params, rng, edges = null) {
  const grad = x.map(() => vec(0, 0));
  const batch = sampleWithoutReplacement(cloud, Math.min(params.sgdBatchSize, cloud.length), rng);
  const batchScale = cloud.length / Math.max(batch.length, 1);

  for (const point of batch) {
    const best = computeNearestCenterIndex(point, x);
    grad[best] = add(grad[best], scale(sub(x[best], point), batchScale));
  }

  for (let i = 0; i < x.length; i += 1) {
    grad[i] = add(grad[i], scale(x[i], 2 * params.lambda));
  }

  const graph = edges ? graphFromEdges(x, edges) : buildKnnGraph(x, 2);
  for (const [a, b] of graph.edges) {
    const delta = sub(x[a], x[b]);
    grad[a] = add(grad[a], scale(delta, 2 * params.mu));
    grad[b] = sub(grad[b], scale(delta, 2 * params.mu));
  }

  const { weights } = assignPoints(x, cloud);
  const totalMass = Math.max(weights.reduce((sum, value) => sum + value, 0), 1);
  const sampledPairs = Math.min(params.sgdPairSamples, (x.length * (x.length - 1)) / 2);
  const pairNormaliser = ((x.length * (x.length - 1)) / 2) / Math.max(sampledPairs, 1);

  for (let sample = 0; sample < sampledPairs; sample += 1) {
    const i = Math.floor(rng() * x.length);
    let j = Math.floor(rng() * (x.length - 1));
    if (j >= i) {
      j += 1;
    }
    const lo = Math.min(i, j);
    const hi = Math.max(i, j);
    const result = dijkstra(graph.adjacency, lo);
    if (!Number.isFinite(result.dist[hi])) {
      continue;
    }
    const coeff = pairNormaliser * 0.5 * (weights[lo] * weights[hi]) / (totalMass * totalMass);
    const path = reconstructPath(result.prev, lo, hi);
    for (let p = 0; p < path.length - 1; p += 1) {
      const a = path[p];
      const b = path[p + 1];
      const delta = sub(x[a], x[b]);
      grad[a] = add(grad[a], scale(delta, 2 * coeff));
      grad[b] = sub(grad[b], scale(delta, 2 * coeff));
    }
  }

  return { grad, graph };
}

function computeNearestCenterIndex(point, centers) {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < centers.length; i += 1) {
    const dx = point.x - centers[i].x;
    const dy = point.y - centers[i].y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDist) {
      bestDist = d2;
      best = i;
    }
  }
  return best;
}

function runSgdIteration(state, graphMode = "knn", pushHistory = true, fixedEdges = null) {
  const previousX = clonePoints(state.x);
  const previousObjective = state.history[state.history.length - 1]?.objective ?? null;
  const graphEdges = fixedEdges !== null ? fixedEdges : graphMode === "fixed" ? state.edges : null;
  const { grad } = computeSgdGradient(state.x, state.cloud, state, state.sgdRng, graphEdges);
  const learningRate = state.sgdLearningRate / Math.sqrt(1 + state.iteration * state.sgdDecay);
  state.x = state.x.map((point, index) => sub(point, scale(grad[index], learningRate)));
  const graph =
    fixedEdges !== null
      ? graphFromEdges(state.x, fixedEdges)
      : graphMode === "fixed"
        ? graphFromEdges(state.x, state.edges)
        : buildKnnGraph(state.x, 2);
  state.edges = graph.edges;
  const metrics = evaluateObjective(state.x, state.edges, state.cloud, state);
  const motion = computeMotionStats(state.x, previousX);
  const objectiveDelta = previousObjective === null ? 0 : Math.abs(metrics.objective - previousObjective);
  state.iteration += 1;
  if (pushHistory) {
    recordHistory(state, metrics, motion.total, objectiveDelta);
  }
  return { metrics, primal: motion.total, dual: objectiveDelta, motion };
}

function runFixedKSgdConvergence(state, options = {}) {
  const {
    maxIterations = 96,
    minIterations = 12,
    pushHistory = true,
    graphMode = "knn",
    maxRuntimeMs = Infinity,
    motionTolerance = 0.0005,
    objectiveTolerance = 0.0005,
  } = options;

  let latest = null;
  let motion = { total: Infinity, avg: Infinity, max: Infinity };
  const fixedEdges = graphMode === "knn" ? buildKnnGraph(state.x, 2).edges : canonicalizeEdges(state.edges);
  state.edges = fixedEdges;
  const startTime = globalThis.performance?.now?.() ?? Date.now();

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    latest = runSgdIteration(state, graphMode, pushHistory, fixedEdges);
    motion = latest.motion;
    const settled = motion.avg <= motionTolerance && latest.dual <= objectiveTolerance;
    if (iteration + 1 >= minIterations && settled) {
      return { ...latest, motion, settled: true, iterations: iteration + 1 };
    }
    if ((globalThis.performance?.now?.() ?? Date.now()) - startTime >= maxRuntimeMs) {
      return { ...latest, motion, settled: false, iterations: iteration + 1, timedOut: true };
    }
  }

  return { ...latest, motion, settled: false, iterations: maxIterations };
}

function runFixedKConvergence(state, options = {}) {
  if (state.optimizer === "sgd") {
    return runFixedKSgdConvergence(state, options);
  }
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

function runConvergedOptimizationWithEdgeOptimization(state, resetAdmmVariables, options = {}) {
  const convergenceOptions = {
    maxIterations: state.optimizer === "sgd" ? 96 : 48,
    minIterations: state.optimizer === "sgd" ? 12 : 6,
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
  if (state.optimizer === "sgd") {
    return runFixedKConvergence(state, {
      maxIterations: 160,
      minIterations: 20,
      graphMode: "knn",
      maxRuntimeMs: 40,
      motionTolerance: 0.00025,
      objectiveTolerance: 0.00025,
    });
  }
  return runFixedKConvergence(state, {
    maxIterations: 96,
    minIterations: 12,
    primalAbsTolerance: 0.00025,
    primalRelTolerance: 0.002,
    graphMode: "knn",
    maxRuntimeMs: 40,
  });
}

export {
  buildCompleteGraph,
  buildKnnGraph,
  canonicalizeEdges,
  computeClusterEnergies,
  computeMotionStats,
  computeResidualThresholds,
  evaluateObjective,
  optimizeEdgesForState,
  proxX,
  recordHistory,
  runAdmmIteration,
  runConvergedOptimizationWithEdgeOptimization,
  runFinalAdmmCompletion,
  runFixedKConvergence,
  runSgdIteration,
};
