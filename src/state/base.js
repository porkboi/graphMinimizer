import { kMeans, sampleInitialHubs, seededPointCloud } from "../core/cloud.js";
import { buildMaxGraph, canonicalizeEdges } from "../core/graph.js";
import { clonePoints, mulberry32, vec } from "../core/math.js";

const sgdDefaults = {
  learningRate: 1e-6,
  decay: 0.04,
  batchSize: 160,
  pairSamples: 12,
};

const admmDefaults = {
  eta: 1e-3,
  decay: 0.04,
};

function optimizerLabel(method) {
  return method === "sgd" ? "SGD" : "ADMM";
}

function syncStateParams(state, ui, optimizationMethod) {
  state.rho = Number(ui.rho.value);
  state.lambda = Number(ui.lambda.value);
  state.mu = Number(ui.mu.value);
  state.stationCost = Number(ui.stationCost.value);
  state.splitTopK = Math.max(1, Math.round(Number(ui.splitTopK.value)));
  state.optimizer = optimizationMethod;
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

function createBaseState(mode, cloud, x, edges, runtime) {
  const state = {
    mode,
    cloud,
    playing: true,
    iteration: 0,
    optimizer: runtime.optimizationMethod,
    lambda: Number(runtime.ui.lambda.value),
    mu: Number(runtime.ui.mu.value),
    rho: Number(runtime.ui.rho.value),
    stationCost: Number(runtime.ui.stationCost.value),
    splitTopK: Math.max(1, Math.round(Number(runtime.ui.splitTopK.value))),
    sgdLearningRate: sgdDefaults.learningRate,
    sgdDecay: sgdDefaults.decay,
    sgdBatchSize: sgdDefaults.batchSize,
    sgdPairSamples: sgdDefaults.pairSamples,
    sgdRng: mulberry32(runtime.currentCloudSeed + x.length * 31 + mode.length * 17),
    history: [],
    lastAction: "Initialised",
    forceStop: false,
    tuningPhase: null,
    tuningPhaseData: null,
    tuningBaseline: null,
    tuningLastConvergence: null,
    tuningAcceptedContext: null,
    admmEta: admmDefaults.eta,
    admmDecay: admmDefaults.decay,
  };
  resetAdmmVariables(state, x, edges);
  return state;
}

function initializeTuningLoopState(state, lastAction) {
  state.outerIteration = 0;
  state.tuningPhase = "cycleStart";
  state.tuningPhaseData = null;
  state.tuningBaseline = null;
  state.tuningLastConvergence = null;
  state.tuningAcceptedContext = null;
  state.forceStop = false;
  state.lastAction = lastAction;
}

function getRuntimeCloud(runtime) {
  return runtime.pointCloudSource?.points ?? seededPointCloud(runtime.currentCloudSeed);
}

function createExplorerState(runtime) {
  const cloud = getRuntimeCloud(runtime);
  const hubCount = Number(runtime.ui.hubCount.value);
  const x = sampleInitialHubs(cloud, hubCount);
  return createBaseState("explorer", cloud, x, [], runtime);
}

function createTuningState(runtime) {
  const cloud = getRuntimeCloud(runtime);
  const startK = 3;
  const clustering = kMeans(cloud, startK, 16, 111);
  const graph = buildMaxGraph(clustering.centers);
  const state = createBaseState("tuning", cloud, clustering.centers, graph.edges, runtime);
  initializeTuningLoopState(state, `Start with ${startK} hubs from KMeans`);
  return state;
}

function createCanvasState(runtime) {
  const points = clonePoints(runtime.customCloud);
  if (points.length === 0) {
    const state = createBaseState("canvas", [], [], [], runtime);
    state.canvasSolveMode = runtime.canvasSolveMode;
    state.outerIteration = 0;
    state.playing = false;
    state.lastAction = "Click inside the canvas to add points";
    return state;
  }

  let x = [];
  let edges = [];
  let lastAction = "";

  if (runtime.canvasSolveMode === "tuning") {
    const startK = Math.max(1, Math.min(3, points.length));
    const clustering = kMeans(points, startK, 16, 111);
    const graph = buildMaxGraph(clustering.centers);
    x = clustering.centers;
    edges = graph.edges;
    lastAction = `Start with ${x.length} hubs from KMeans on ${points.length} clicked points`;
  } else {
    const hubCount = Math.max(1, Math.min(Number(runtime.ui.hubCount.value), points.length));
    x = sampleInitialHubs(points, hubCount, 1337);
    lastAction = `Seeded ${x.length} hubs from ${points.length} clicked points for ${optimizerLabel(runtime.optimizationMethod)}`;
  }

  const state = createBaseState("canvas", points, x, edges, runtime);
  state.canvasSolveMode = runtime.canvasSolveMode;
  if (runtime.canvasSolveMode === "tuning") {
    initializeTuningLoopState(state, lastAction);
  } else {
    state.outerIteration = 0;
    state.lastAction = lastAction;
  }
  return state;
}

export {
  createBaseState,
  createCanvasState,
  createExplorerState,
  createTuningState,
  initializeTuningLoopState,
  optimizerLabel,
  resetAdmmVariables,
  sgdDefaults,
  syncStateParams,
};
