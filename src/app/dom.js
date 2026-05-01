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
  primalMetricLabel: document.getElementById("primalMetricLabel"),
  dualMetricLabel: document.getElementById("dualMetricLabel"),
  weightLabel: document.getElementById("weightLabel"),
  weightValue: document.getElementById("weightValue"),
  pathLabel: document.getElementById("pathLabel"),
  pathValue: document.getElementById("pathValue"),
  playPause: document.getElementById("playPause"),
  stepButton: document.getElementById("stepButton"),
  resetButton: document.getElementById("resetButton"),
  randomizeButton: document.getElementById("randomizeButton"),
  voronoiToggle: document.getElementById("voronoiToggle"),
  citySimulation: document.getElementById("citySimulation"),
  uniform256: document.getElementById("uniform256"),
  uniform768: document.getElementById("uniform768"),
  uniform1536: document.getElementById("uniform1536"),
  motionToggle: document.getElementById("motionToggle"),
  optimizerAdmm: document.getElementById("optimizerAdmm"),
  optimizerSgd: document.getElementById("optimizerSgd"),
  canvasSolverGroup: document.getElementById("canvasSolverGroup"),
  canvasExplorerMode: document.getElementById("canvasExplorerMode"),
  canvasTuningMode: document.getElementById("canvasTuningMode"),
  skipAdmmTuningAnimationGroup: document.getElementById("skipAdmmTuningAnimationGroup"),
  skipAdmmTuningAnimation: document.getElementById("skipAdmmTuningAnimation"),
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
  splitTopKGroup: document.getElementById("splitTopKGroup"),
  splitTopK: document.getElementById("splitTopK"),
  splitTopKValue: document.getElementById("splitTopKValue"),
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
  explorer: {
    admm:
      'The browser runs a 2D, deterministic ADMM-style solver over hub locations <code>x, z1, z2, z3, u1, u2, u3</code>. Each frame updates point assignments, shortest-path regularisation, quadratic shrinkage, graph total variation, and a station-count penalty.',
    sgd:
      "The browser runs a 2D stochastic-gradient descent path over the same objective. Each step samples point and path terms, rebuilds the local kNN graph, and descends the combined fit, smoothness, shrinkage, graph variation, and station-cost energy.",
  },
  tuning: {
    admm:
      "This tab reuses the explorer's local kNN-graph ADMM solver at each fixed hub count. It starts from 3 KMeans hubs, settles the current geometry with the same ADMM updates as the explorer, then splits the highest-energy Voronoi cell and keeps the new hub only when that locally converged split lowers the loss after the station penalty is included.",
    sgd:
      "This tab reuses the explorer's local kNN-graph SGD solver at each fixed hub count. It starts from 3 KMeans hubs, settles the current geometry with the same SGD updates as the explorer, then splits the highest-energy Voronoi cell and keeps the new hub only when that locally converged split lowers the loss after the station penalty is included.",
  },
  canvas: {
    admm:
      "This tab turns the main canvas into a point-cloud editor. Each click adds an observed point, then the browser re-seeds hub locations from that clicked cloud and solves with the selected modular solver path: the explorer's local kNN-graph ADMM update or the existing hub-tuning loop.",
    sgd:
      "This tab turns the main canvas into a point-cloud editor. Each click adds an observed point, then the browser re-seeds hub locations from that clicked cloud and solves with the selected modular solver path: the explorer's local kNN-graph SGD update or the existing hub-tuning loop.",
  },
};

const sliderConfigs = [
  { input: ui.hubCount, value: ui.hubCountValue, format: (raw) => String(Math.round(Number(raw))) },
  { input: ui.rho, value: ui.rhoValue, format: (raw) => Number(raw).toFixed(2) },
  { input: ui.lambda, value: ui.lambdaValue, format: (raw) => Number(raw).toFixed(2) },
  { input: ui.mu, value: ui.muValue, format: (raw) => Number(raw).toFixed(2) },
  { input: ui.stationCost, value: ui.stationCostValue, format: (raw) => Number(raw).toFixed(2) },
  { input: ui.splitTopK, value: ui.splitTopKValue, format: (raw) => String(Math.round(Number(raw))) },
];

export {
  modeCopy,
  objectiveCanvas,
  objectiveCtx,
  residualCanvas,
  residualCtx,
  sceneCanvas,
  sceneCtx,
  sliderConfigs,
  ui,
};
