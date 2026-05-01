import { kMeans } from "../core/cloud.js";
import { canonicalizeEdges } from "../core/graph.js";
import { computeClusterEnergies, evaluateObjective } from "../core/objective.js";
import { clonePoints } from "../core/math.js";
import {
  buildKnnGraph,
  computeMotionStats,
  computeResidualThresholds,
  optimizeEdgesForState,
  recordHistory,
  runAdmmIteration,
  runConvergedOptimizationWithEdgeOptimization,
  runFinalAdmmCompletion,
  runSgdIteration,
  runFixedStructureAdmmBlock,
} from "../solver/optimizers.js";
import { createBaseState, initializeTuningLoopState, optimizerLabel, resetAdmmVariables, syncStateParams } from "./base.js";

function shouldAnimateTuning(state, skipAdmmTuningAnimation) {
  return state.optimizer === "admm" && !skipAdmmTuningAnimation;
}

function createSimulationRuntime(baseState) {
  return {
    currentCloudSeed: 0,
    customCloud: [],
    canvasSolveMode: "explorer",
    optimizationMethod: baseState.optimizer,
    ui: {
      lambda: { value: String(baseState.lambda) },
      mu: { value: String(baseState.mu) },
      rho: { value: String(baseState.rho) },
      stationCost: { value: String(baseState.stationCost) },
      splitTopK: { value: String(baseState.splitTopK ?? 1) },
    },
  };
}

function simulateFixedHubRefinement(baseState, x, edges) {
  const state = createBaseState("simulation", baseState.cloud, x, edges, createSimulationRuntime(baseState));
  state.lambda = baseState.lambda;
  state.mu = baseState.mu;
  state.rho = baseState.rho;
  state.stationCost = baseState.stationCost;
  state.splitTopK = baseState.splitTopK;
  const convergence = runConvergedOptimizationWithEdgeOptimization(state, resetAdmmVariables, {
    pushHistory: false,
    graphMode: "knn",
  });
  return { state, metrics: convergence.metrics, convergence };
}

function simulateFinalCompletion(baseState, x, edges) {
  const state = createBaseState("simulation", baseState.cloud, x, edges, createSimulationRuntime(baseState));
  state.lambda = baseState.lambda;
  state.mu = baseState.mu;
  state.rho = baseState.rho;
  state.stationCost = baseState.stationCost;
  state.splitTopK = baseState.splitTopK;
  const completion = runFinalAdmmCompletion(state);
  return { state, metrics: completion.metrics, completion };
}

function getTuningConvergenceOptions(state, mode, graphMode) {
  if (mode === "final") {
    if (state.optimizer === "sgd") {
      return {
        maxIterations: 160,
        minIterations: 20,
        graphMode,
        motionTolerance: 0.00025,
        objectiveTolerance: 0.00025,
      };
    }
    return {
      maxIterations: 96,
      minIterations: 12,
      primalAbsTolerance: 0.00025,
      primalRelTolerance: 0.002,
      graphMode,
    };
  }

  if (state.optimizer === "sgd") {
    return {
      maxIterations: 96,
      minIterations: 12,
      graphMode,
      motionTolerance: 0.0005,
      objectiveTolerance: 0.0005,
    };
  }
  return {
    maxIterations: 48,
    minIterations: 6,
    primalAbsTolerance: 0.0005,
    primalRelTolerance: 0.004,
    graphMode,
  };
}

function startTuningConvergencePhase(state, phaseName, mode, graphMode, nextPhase, extra = {}) {
  const options = getTuningConvergenceOptions(state, mode, graphMode);
  const fixedEdges = graphMode === "knn" ? buildKnnGraph(state.x, 2).edges : canonicalizeEdges(state.edges);
  state.edges = fixedEdges;
  state.tuningPhase = phaseName;
  state.tuningPhaseData = {
    phaseName,
    mode,
    graphMode,
    nextPhase,
    fixedEdges,
    iterations: 0,
    ...options,
    ...extra,
  };
}

function beginTuningCycle(state) {
  state.tuningBaseline = null;
  state.tuningLastConvergence = null;
  state.tuningAcceptedContext = null;
  startTuningConvergencePhase(state, "settling", "regular", "knn", "evaluateSplit");
  state.lastAction = `Outer step ${state.outerIteration + 1}: settling ${optimizerLabel(state.optimizer)} with ${state.x.length} hubs`;
}

function runSingleTuningConvergenceIteration(state) {
  const data = state.tuningPhaseData;
  if (!data) {
    return null;
  }

  let latest;
  let motion = null;
  if (state.optimizer === "sgd") {
    latest = runSgdIteration(state, data.graphMode, false, data.fixedEdges);
    motion = latest.motion;
  } else {
    const previousX = clonePoints(state.x);
    latest = runAdmmIteration(state, data.graphMode, false, data.fixedEdges);
    motion = computeMotionStats(state.x, previousX);
  }

  data.iterations += 1;
  let settled = false;
  if (state.optimizer === "sgd") {
    settled = motion.avg <= data.motionTolerance && latest.dual <= data.objectiveTolerance;
  } else {
    const thresholds = computeResidualThresholds(state, data.primalAbsTolerance, data.primalRelTolerance);
    settled = latest.primal <= thresholds.primal && latest.dual <= thresholds.dual;
  }

  const finished = data.iterations >= data.maxIterations || (data.iterations >= data.minIterations && settled);
  const result = {
    ...latest,
    motion,
    settled,
    iterations: data.iterations,
    timedOut: false,
  };

  state.tuningLastConvergence = result;
  return { result, finished };
}

function commitAcceptedTuningState(state, result) {
  if (!result?.metrics) {
    return;
  }
  recordHistory(state, result.metrics, result.primal, result.dual);
}

function captureTuningBaseline(state, baselineFinalObjective) {
  state.tuningBaseline = {
    x: clonePoints(state.x),
    edges: canonicalizeEdges(state.edges),
    iteration: state.iteration,
    weights: state.weights.slice(),
    highlightedPath: [...state.highlightedPath],
    highlightedPair: [...state.highlightedPair],
    historyLength: state.history.length,
    finalObjective: baselineFinalObjective,
  };
}

function restoreTuningBaseline(state) {
  const baseline = state.tuningBaseline;
  if (!baseline) {
    return;
  }
  resetAdmmVariables(state, baseline.x, baseline.edges);
  state.iteration = baseline.iteration;
  state.weights = baseline.weights;
  state.highlightedPath = baseline.highlightedPath;
  state.highlightedPair = baseline.highlightedPair;
  state.history.length = baseline.historyLength;
}

function advanceTuningConvergencePhase(state) {
  const step = runSingleTuningConvergenceIteration(state);
  if (!step || !step.finished) {
    return;
  }

  const data = state.tuningPhaseData;
  const edgeOptimization =
    data.mode === "regular" && !data.edgeOptimized
      ? optimizeEdgesForState(state, step.result.metrics)
      : { improved: false, removedEdges: 0 };

  if (data.mode === "regular" && edgeOptimization.improved) {
    resetAdmmVariables(state, state.x, edgeOptimization.edges);
    startTuningConvergencePhase(state, data.phaseName ?? "settlingFixed", data.mode, "fixed", data.nextPhase, {
      edgeOptimized: true,
      removedEdges: edgeOptimization.removedEdges,
    });
    state.lastAction = `Outer step ${state.outerIteration + 1}: pruned ${edgeOptimization.removedEdges} edges, continuing ${optimizerLabel(state.optimizer)} settle`;
    return;
  }

  state.tuningLastConvergence = {
    ...step.result,
    edgeOptimization:
      data.mode === "regular"
        ? edgeOptimization
        : { improved: false, removedEdges: data.removedEdges ?? 0 },
  };
  if (data.nextPhase === "evaluateSplit") {
    commitAcceptedTuningState(state, state.tuningLastConvergence);
  }
  state.tuningPhase = data.nextPhase;
  state.tuningPhaseData = null;
}

function buildSplitProposal(state) {
  const currentEval = evaluateObjective(state.x, state.edges, state.cloud, state);
  const clusterEnergies = computeClusterEnergies(currentEval.assignments, state.x);
  const candidateClusters = [];
  for (let i = 0; i < clusterEnergies.length; i += 1) {
    if (currentEval.assignments[i].length < 2) {
      continue;
    }
    candidateClusters.push({ clusterIndex: i, clusterEnergy: clusterEnergies[i] });
  }
  if (candidateClusters.length === 0) {
    return null;
  }

  candidateClusters.sort((a, b) => b.clusterEnergy - a.clusterEnergy);
  const topClusters = candidateClusters.slice(0, Math.max(1, state.splitTopK ?? 1));

  let best = null;
  for (const { clusterIndex, clusterEnergy } of topClusters) {
    const clusterPoints = currentEval.assignments[clusterIndex];
    const splitSeed = state.outerIteration + clusterIndex * 19 + state.x.length * 7;
    const split = kMeans(clusterPoints, 2, 12, splitSeed);
    if (split.centers.length < 2) {
      continue;
    }

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
            clusterEnergy,
            candidate,
            anchorIndex,
            simulation,
          };
        }
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

function stepExplorerState(state, runtime) {
  if (state.x.length === 0 || state.cloud.length === 0) {
    state.playing = false;
    return;
  }
  syncStateParams(state, runtime.ui, runtime.optimizationMethod);
  if (state.optimizer === "sgd") {
    runSgdIteration(state, "knn", true);
  } else {
    runFixedStructureAdmmBlock(state, 4, true);
  }
}

function stepTuningState(state, runtime) {
  if (state.x.length === 0 || state.cloud.length === 0) {
    state.playing = false;
    return;
  }
  syncStateParams(state, runtime.ui, runtime.optimizationMethod);
  const solverName = optimizerLabel(state.optimizer);
  const animateSolver = shouldAnimateTuning(state, runtime.skipAdmmTuningAnimation);

  if (!state.tuningPhase) {
    initializeTuningLoopState(state, `Start tuning with ${state.x.length} hubs`);
  }

  while (true) {
    if (state.tuningPhase === "cycleStart") {
      beginTuningCycle(state);
      if (animateSolver) {
        return;
      }
      continue;
    }

    if (state.tuningPhase === "settling" || state.tuningPhase === "settlingFixed" || state.tuningPhase === "finalizing") {
      if (animateSolver) {
        advanceTuningConvergencePhase(state);
        return;
      }
      while (state.tuningPhase === "settling" || state.tuningPhase === "settlingFixed" || state.tuningPhase === "finalizing") {
        advanceTuningConvergencePhase(state);
      }
      continue;
    }

    if (state.tuningPhase === "evaluateSplit") {
      const baselineFinal = simulateFinalCompletion(state, clonePoints(state.x), canonicalizeEdges(state.edges));
      captureTuningBaseline(state, baselineFinal.metrics.objective);
      state.outerIteration += 1;

      const splitProposal = buildSplitProposal(state);
      if (splitProposal && splitProposal.delta > 0) {
        const acceptedCompletion = simulateFinalCompletion(
          splitProposal.simulation.state,
          clonePoints(splitProposal.simulation.state.x),
          canonicalizeEdges(splitProposal.simulation.state.edges),
        );
        const baselineFinalObjective = state.tuningBaseline?.finalObjective ?? Infinity;
        if (acceptedCompletion.metrics.objective + 1e-6 < baselineFinalObjective) {
          adoptSimulationState(
            state,
            acceptedCompletion,
            `Added hub from Voronoi split ${splitProposal.clusterIndex} via ${splitProposal.anchorIndex}; final dF=${(
              baselineFinalObjective - acceptedCompletion.metrics.objective
            ).toFixed(3)}`,
          );
          commitAcceptedTuningState(state, acceptedCompletion.completion);
          state.tuningPhase = "cycleStart";
          state.tuningPhaseData = null;
          state.tuningBaseline = null;
          state.tuningAcceptedContext = null;
          return;
        }

        state.tuningAcceptedContext = {
          rejectedObjective: acceptedCompletion.metrics.objective,
          baselineFinalObjective,
        };
        startTuningConvergencePhase(state, "finalizing", "final", "knn", "completeRejectedSplit");
        state.lastAction = `Rejected hub split at outer step ${state.outerIteration}; final ${solverName} objective rose from ${baselineFinalObjective.toFixed(3)} to ${acceptedCompletion.metrics.objective.toFixed(3)}; running final ${solverName} cleanup`;
        if (animateSolver) {
          return;
        }
        continue;
      }

      startTuningConvergencePhase(state, "finalizing", "final", "knn", "completeNoSplit");
      state.lastAction = `Outer step ${state.outerIteration}: no improving split, running final ${solverName} cleanup`;
      if (animateSolver) {
        return;
      }
      continue;
    }

    if (state.tuningPhase === "prepareAcceptedFinalization") {
      state.tuningAcceptedContext = {
        edgeOptimization: state.tuningLastConvergence?.edgeOptimization ?? { improved: false, removedEdges: 0 },
      };
      startTuningConvergencePhase(state, "finalizing", "final", "knn", "cycleAcceptedSplit");
      state.lastAction = `${state.lastAction}; running final ${solverName} cleanup`;
      if (animateSolver) {
        return;
      }
      continue;
    }

    if (state.tuningPhase === "completeNoSplit") {
      const completion = state.tuningLastConvergence;
      commitAcceptedTuningState(state, completion);
      const settleLabel = completion?.settled ? "settled" : `hit the ${solverName} step cap`;
      const completionLabel = completion?.settled
        ? `final ${solverName} completed in ${completion.iterations} steps`
        : `final ${solverName} hit the step cap after ${completion?.iterations ?? 0} steps`;
      state.lastAction = `No improving hub split after ${settleLabel} at outer step ${state.outerIteration}; ${completionLabel}`;
      state.tuningPhase = "done";
      state.playing = false;
      return;
    }

    if (state.tuningPhase === "completeRejectedSplit") {
      const completion = state.tuningLastConvergence;
      commitAcceptedTuningState(state, completion);
      const completionLabel = completion?.settled
        ? `final ${solverName} completed in ${completion.iterations} steps`
        : `final ${solverName} hit the step cap after ${completion?.iterations ?? 0} steps`;
      const baselineFinalObjective = state.tuningAcceptedContext?.baselineFinalObjective ?? Infinity;
      const rejectedObjective = state.tuningAcceptedContext?.rejectedObjective ?? Infinity;
      state.lastAction = `Rejected hub split at outer step ${state.outerIteration}; final ${solverName} objective rose from ${baselineFinalObjective.toFixed(3)} to ${rejectedObjective.toFixed(3)}; ${completionLabel}`;
      state.tuningPhase = "done";
      state.tuningPhaseData = null;
      state.tuningAcceptedContext = null;
      state.playing = false;
      return;
    }

    if (state.tuningPhase === "cycleAcceptedSplit") {
      const completion = state.tuningLastConvergence;
      const baselineFinalObjective = state.tuningBaseline?.finalObjective ?? Infinity;
      if (completion.metrics.objective + 1e-6 < baselineFinalObjective || state.forceStop) {
        commitAcceptedTuningState(state, completion);
        const edgeNote = state.tuningAcceptedContext?.edgeOptimization?.improved
          ? `; pruned ${state.tuningAcceptedContext.edgeOptimization.removedEdges} edges after convergence`
          : "";
        const finalDelta = baselineFinalObjective - completion.metrics.objective;
        const completionLabel = completion.settled
          ? `final ${solverName} completed in ${completion.iterations} steps`
          : `final ${solverName} hit the step cap after ${completion.iterations} steps`;
        state.lastAction = `${state.lastAction}; ${completionLabel}; final dF=${finalDelta.toFixed(3)}${edgeNote}`;
        state.tuningPhase = "cycleStart";
        state.tuningPhaseData = null;
        state.tuningBaseline = null;
        state.tuningAcceptedContext = null;
        return;
      }

      restoreTuningBaseline(state);
      state.lastAction = `Rejected hub split at outer step ${state.outerIteration}; final ${solverName} objective rose from ${baselineFinalObjective.toFixed(3)} to ${completion.metrics.objective.toFixed(3)}`;
      state.forceStop = true;
      state.tuningPhase = "done";
      state.tuningPhaseData = null;
      state.tuningAcceptedContext = null;
      state.playing = false;
      return;
    }

    if (state.tuningPhase === "done") {
      state.playing = false;
      return;
    }
  }
}

export { stepExplorerState, stepTuningState };
