import { defaultCloudBounds, randomSeed, seededPointCloud, uniformPointCloud } from "../core/cloud.js";
import { clamp } from "../core/math.js";
import { drawChart, drawScene, camera, getSceneBounds, unprojectPoint } from "../render/canvas.js";
import { createCanvasState, createExplorerState, createTuningState, optimizerLabel } from "../state/base.js";
import { stepExplorerState, stepTuningState } from "../state/tuning.js";
import {
  modeCopy,
  objectiveCanvas,
  objectiveCtx,
  residualCanvas,
  residualCtx,
  sceneCanvas,
  sceneCtx,
  sliderConfigs,
  ui,
} from "./dom.js";

function createRuntime() {
  return {
    activeMode: "explorer",
    currentCloudSeed: 12,
    customCloud: [],
    pointCloudSource: {
      type: "seeded",
      label: "Seeded clusters",
      seed: 12,
      points: seededPointCloud(12),
    },
    motionEnabled: false,
    motionStepMs: 150,
    pointVelocities: [],
    canvasSolveMode: "explorer",
    optimizationMethod: "admm",
    showVoronoiOverlay: false,
    skipAdmmTuningAnimation: false,
    explorerState: null,
    tuningState: null,
    canvasState: null,
    ui,
  };
}

function buildPointVelocities(points, seed = randomSeed()) {
  let state = seed >>> 0;
  const nextRandom = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return points.map(() => {
    const angle = nextRandom() * Math.PI * 2;
    const speed = 0.004 + nextRandom() * 0.012;
    return { x: Math.cos(angle) * speed, y: Math.sin(angle) * speed };
  });
}

function syncGeneratedSourceState(runtime) {
  hydrateStates(runtime);
  if (runtime.activeMode !== "canvas") {
    resetState(runtime, runtime.activeMode);
    return;
  }
  updateUi(runtime);
}

function setPointCloudSource(runtime, source) {
  runtime.pointCloudSource = source;
  runtime.pointVelocities = buildPointVelocities(source.points, source.seed + source.points.length * 13);
  if (runtime.activeMode === "canvas") {
    runtime.activeMode = "explorer";
  }
  syncGeneratedSourceState(runtime);
}

function spawnUniformCloud(runtime, count) {
  const seed = randomSeed();
  setPointCloudSource(runtime, {
    type: "uniform",
    label: `Uniform ${count}`,
    seed,
    points: uniformPointCloud(count, seed, defaultCloudBounds),
  });
}

function resetSeededCloud(runtime) {
  const seed = randomSeed();
  runtime.currentCloudSeed = seed;
  setPointCloudSource(runtime, {
    type: "seeded",
    label: "Seeded clusters",
    seed,
    points: seededPointCloud(seed),
  });
}

function advanceGeneratedCloudMotion(runtime) {
  if (!runtime.motionEnabled || runtime.activeMode === "canvas" || !runtime.pointCloudSource?.points?.length) {
    return;
  }

  const points = runtime.pointCloudSource.points;
  if (runtime.pointVelocities.length !== points.length) {
    runtime.pointVelocities = buildPointVelocities(points, runtime.pointCloudSource.seed + points.length * 13);
  }

  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const velocity = runtime.pointVelocities[i];
    point.x += velocity.x;
    point.y += velocity.y;

    if (point.x <= defaultCloudBounds.minX || point.x >= defaultCloudBounds.maxX) {
      velocity.x *= -1;
      point.x = clamp(point.x, defaultCloudBounds.minX, defaultCloudBounds.maxX);
    }
    if (point.y <= defaultCloudBounds.minY || point.y >= defaultCloudBounds.maxY) {
      velocity.y *= -1;
      point.y = clamp(point.y, defaultCloudBounds.minY, defaultCloudBounds.maxY);
    }
  }
}

function getActiveState(runtime) {
  if (runtime.activeMode === "explorer") {
    return runtime.explorerState;
  }
  if (runtime.activeMode === "tuning") {
    return runtime.tuningState;
  }
  return runtime.canvasState;
}

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

function hydrateStates(runtime) {
  runtime.explorerState = createExplorerState(runtime);
  runtime.tuningState = createTuningState(runtime);
  runtime.canvasState = createCanvasState(runtime);
}

function stepExplorer(runtime) {
  stepExplorerState(runtime.explorerState, runtime);
}

function stepTuning(runtime) {
  stepTuningState(runtime.tuningState, runtime);
}

function stepCanvas(runtime) {
  if (runtime.canvasSolveMode === "tuning") {
    stepTuningState(runtime.canvasState, runtime);
  } else {
    stepExplorerState(runtime.canvasState, runtime);
  }
}

function stepActive(runtime) {
  if (runtime.activeMode === "explorer") {
    stepExplorer(runtime);
  } else if (runtime.activeMode === "tuning") {
    stepTuning(runtime);
  } else {
    stepCanvas(runtime);
  }
}

function updateUi(runtime) {
  const state = getActiveState(runtime);
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

  const showOuterIteration = state.mode === "tuning" || (state.mode === "canvas" && runtime.canvasSolveMode === "tuning");
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
  ui.primalMetricLabel.textContent = state.optimizer === "sgd" ? "Hub motion" : "Primal residual";
  ui.dualMetricLabel.textContent = state.optimizer === "sgd" ? "Objective delta" : "Dual residual";
  syncSliderLabels();
  ui.modeSummary.innerHTML = modeCopy[runtime.activeMode][runtime.optimizationMethod];
  ui.explorerTab.textContent = `${optimizerLabel(runtime.optimizationMethod)} Explorer`;
  ui.explorerTab.classList.toggle("is-active", runtime.activeMode === "explorer");
  ui.tuningTab.classList.toggle("is-active", runtime.activeMode === "tuning");
  ui.canvasTab.classList.toggle("is-active", runtime.activeMode === "canvas");
  ui.optimizerAdmm.classList.toggle("is-active", runtime.optimizationMethod === "admm");
  ui.optimizerSgd.classList.toggle("is-active", runtime.optimizationMethod === "sgd");
  ui.canvasSolverGroup.classList.toggle("is-hidden", runtime.activeMode !== "canvas");
  ui.canvasExplorerMode.textContent = `${optimizerLabel(runtime.optimizationMethod)} Explorer`;
  ui.canvasExplorerMode.classList.toggle("is-active", runtime.canvasSolveMode === "explorer");
  ui.canvasTuningMode.classList.toggle("is-active", runtime.canvasSolveMode === "tuning");
  ui.skipAdmmTuningAnimation.checked = runtime.skipAdmmTuningAnimation;
  ui.skipAdmmTuningAnimationGroup.classList.toggle(
    "is-hidden",
    runtime.optimizationMethod !== "admm" ||
      (runtime.activeMode !== "tuning" && !(runtime.activeMode === "canvas" && runtime.canvasSolveMode === "tuning")),
  );
  ui.splitTopKGroup.classList.toggle(
    "is-hidden",
    runtime.activeMode !== "tuning" && !(runtime.activeMode === "canvas" && runtime.canvasSolveMode === "tuning"),
  );
  ui.hubCountGroup.classList.toggle(
    "is-hidden",
    runtime.activeMode === "tuning" || (runtime.activeMode === "canvas" && runtime.canvasSolveMode === "tuning"),
  );
  ui.randomizeButton.textContent = runtime.activeMode === "canvas" ? "Clear Canvas" : "Randomise";
  ui.voronoiToggle.textContent = runtime.showVoronoiOverlay ? "Hide Voronoi" : "Show Voronoi";
  ui.voronoiToggle.classList.toggle("is-active", runtime.showVoronoiOverlay);
  ui.motionToggle.textContent = runtime.motionEnabled ? "Stop Motion" : "Start Motion";
  ui.motionToggle.classList.toggle("is-active", runtime.motionEnabled);
  ui.citySimulation.classList.toggle("is-active", runtime.pointCloudSource?.type === "seeded");
  ui.uniform256.classList.toggle("is-active", runtime.pointCloudSource?.type === "uniform" && runtime.pointCloudSource.points.length === 256);
  ui.uniform768.classList.toggle("is-active", runtime.pointCloudSource?.type === "uniform" && runtime.pointCloudSource.points.length === 768);
  ui.uniform1536.classList.toggle("is-active", runtime.pointCloudSource?.type === "uniform" && runtime.pointCloudSource.points.length === 1536);
  sceneCanvas.classList.toggle("is-editable", runtime.activeMode === "canvas");

  if (state.mode === "tuning") {
    ui.weightLabel.textContent = "Hub / edge count";
    ui.weightValue.textContent = `${current.hubCount} / ${current.edgeCount}`;
    ui.pathLabel.textContent = "Last tuning action";
    ui.pathValue.textContent = state.lastAction;
  } else if (state.mode === "canvas") {
    ui.weightLabel.textContent = "Point / hub count";
    ui.weightValue.textContent = `${state.cloud.length} / ${state.x.length}`;
    ui.pathLabel.textContent = runtime.canvasSolveMode === "tuning" ? "Canvas tuning action" : "Canvas action";
    ui.pathValue.textContent = state.lastAction;
  } else {
    ui.weightLabel.textContent = "Largest hub weight";
    ui.weightValue.textContent = String(current.maxWeight);
    ui.pathLabel.textContent = "Cloud / path";
    ui.pathValue.textContent = `${runtime.pointCloudSource?.label ?? "Seeded"}; ${state.highlightedPair[0]} -> ${state.highlightedPair[1]}`;
  }

  drawScene(state, { sceneCanvas, sceneCtx, showVoronoiOverlay: runtime.showVoronoiOverlay });
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
    state.optimizer === "sgd" ? ["motion", "delta"] : ["primal", "dual"],
  );
}

function resetState(runtime, mode = runtime.activeMode) {
  if (mode === "explorer") {
    runtime.explorerState = createExplorerState(runtime);
    for (let i = 0; i < 3; i += 1) {
      stepExplorer(runtime);
    }
  } else if (mode === "tuning") {
    runtime.tuningState = createTuningState(runtime);
    for (let i = 0; i < 2; i += 1) {
      stepTuning(runtime);
    }
  } else {
    runtime.canvasState = createCanvasState(runtime);
    if (runtime.canvasState.x.length > 0) {
      const warmupSteps = runtime.canvasSolveMode === "tuning" ? 2 : 3;
      for (let i = 0; i < warmupSteps; i += 1) {
        stepCanvas(runtime);
      }
    }
  }
  updateUi(runtime);
}

function randomizeCloud(runtime, mode = runtime.activeMode) {
  if (mode === "canvas") {
    runtime.customCloud = [];
    runtime.canvasState = createCanvasState(runtime);
    updateUi(runtime);
    return;
  }
  if (runtime.pointCloudSource?.type === "uniform") {
    spawnUniformCloud(runtime, runtime.pointCloudSource.points.length);
    return;
  }
  resetSeededCloud(runtime);
}

function switchMode(runtime, mode) {
  runtime.activeMode = mode;
  if (getActiveState(runtime).history.length === 0) {
    resetState(runtime, mode);
    return;
  }
  updateUi(runtime);
}

function setOptimizationMethod(runtime, method) {
  if (runtime.optimizationMethod === method) {
    return;
  }
  runtime.optimizationMethod = method;
  hydrateStates(runtime);
  resetState(runtime, runtime.activeMode);
}

function nudgeCamera(runtime, dx = 0, dy = 0) {
  camera.offsetX += dx;
  camera.offsetY += dy;
  updateUi(runtime);
}

function changeZoom(runtime, factor) {
  camera.zoom = clamp(camera.zoom * factor, 0.65, 3.5);
  updateUi(runtime);
}

function bindEvents(runtime) {
  ui.playPause.addEventListener("click", () => {
    const state = getActiveState(runtime);
    state.playing = !state.playing;
    updateUi(runtime);
  });

  ui.stepButton.addEventListener("click", () => {
    const state = getActiveState(runtime);
    state.playing = false;
    stepActive(runtime);
    updateUi(runtime);
  });

  ui.resetButton.addEventListener("click", () => resetState(runtime, runtime.activeMode));
  ui.randomizeButton.addEventListener("click", () => randomizeCloud(runtime, runtime.activeMode));
  ui.voronoiToggle.addEventListener("click", () => {
    runtime.showVoronoiOverlay = !runtime.showVoronoiOverlay;
    updateUi(runtime);
  });
  ui.citySimulation.addEventListener("click", () => resetSeededCloud(runtime));
  ui.uniform256.addEventListener("click", () => spawnUniformCloud(runtime, 256));
  ui.uniform768.addEventListener("click", () => spawnUniformCloud(runtime, 768));
  ui.uniform1536.addEventListener("click", () => spawnUniformCloud(runtime, 1536));
  ui.motionToggle.addEventListener("click", () => {
    runtime.motionEnabled = !runtime.motionEnabled;
    updateUi(runtime);
  });
  ui.optimizerAdmm.addEventListener("click", () => setOptimizationMethod(runtime, "admm"));
  ui.optimizerSgd.addEventListener("click", () => setOptimizationMethod(runtime, "sgd"));
  ui.explorerTab.addEventListener("click", () => switchMode(runtime, "explorer"));
  ui.tuningTab.addEventListener("click", () => switchMode(runtime, "tuning"));
  ui.canvasTab.addEventListener("click", () => switchMode(runtime, "canvas"));
  ui.canvasExplorerMode.addEventListener("click", () => {
    runtime.canvasSolveMode = "explorer";
    runtime.canvasState = createCanvasState(runtime);
    resetState(runtime, runtime.activeMode);
  });
  ui.canvasTuningMode.addEventListener("click", () => {
    runtime.canvasSolveMode = "tuning";
    runtime.canvasState = createCanvasState(runtime);
    resetState(runtime, runtime.activeMode);
  });
  ui.skipAdmmTuningAnimation.addEventListener("change", () => {
    runtime.skipAdmmTuningAnimation = ui.skipAdmmTuningAnimation.checked;
    updateUi(runtime);
  });
  ui.zoomIn.addEventListener("click", () => changeZoom(runtime, 1.2));
  ui.zoomOut.addEventListener("click", () => changeZoom(runtime, 1 / 1.2));
  ui.panUp.addEventListener("click", () => nudgeCamera(runtime, 0, 34));
  ui.panDown.addEventListener("click", () => nudgeCamera(runtime, 0, -34));
  ui.panLeft.addEventListener("click", () => nudgeCamera(runtime, 34, 0));
  ui.panRight.addEventListener("click", () => nudgeCamera(runtime, -34, 0));
  sceneCanvas.addEventListener("click", (event) => {
    if (runtime.activeMode !== "canvas") {
      return;
    }
    const rect = sceneCanvas.getBoundingClientRect();
    const state = getActiveState(runtime);
    const bounds = getSceneBounds(state);
    const screenPoint = {
      x: ((event.clientX - rect.left) / rect.width) * sceneCanvas.width,
      y: ((event.clientY - rect.top) / rect.height) * sceneCanvas.height,
    };
    runtime.customCloud.push(unprojectPoint(screenPoint, bounds, { width: sceneCanvas.width, height: sceneCanvas.height }));
    runtime.canvasState = createCanvasState(runtime);
    runtime.canvasState.playing = true;
    runtime.canvasState.lastAction =
      runtime.canvasSolveMode === "tuning"
        ? `Added point ${runtime.customCloud.length}; hub tuning restarted with ${runtime.canvasState.x.length} hubs`
        : `Added point ${runtime.customCloud.length}; solving with ${runtime.canvasState.x.length} hubs via ${optimizerLabel(runtime.optimizationMethod)}`;
    updateUi(runtime);
  });

  for (const slider of [ui.hubCount, ui.rho, ui.lambda, ui.mu, ui.stationCost, ui.splitTopK]) {
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
    hydrateStates(runtime);
    resetState(runtime, runtime.activeMode);
  });
}

function startAnimation(runtime) {
  let lastTick = 0;
  function animate(timestamp) {
    if (timestamp - lastTick > runtime.motionStepMs) {
      const state = getActiveState(runtime);
      advanceGeneratedCloudMotion(runtime);
      if (state.playing) {
        stepActive(runtime);
      }
      updateUi(runtime);
      lastTick = timestamp;
    }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

function startApp() {
  const runtime = createRuntime();
  hydrateStates(runtime);
  bindEvents(runtime);
  resetState(runtime, "explorer");
  startAnimation(runtime);
}

export { startApp };
