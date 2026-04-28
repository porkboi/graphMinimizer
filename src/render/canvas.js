import { assignPoints, computeClusterEnergies, computeVoronoiCells, polygonCentroid } from "../core/objective.js";
import { TAU, clamp, vec } from "../core/math.js";

const camera = {
  zoom: 1,
  offsetX: 0,
  offsetY: 0,
};

function projectPoint(point, bounds, size, view = camera) {
  const pad = 46;
  const sx = (size.width - pad * 2) / (bounds.maxX - bounds.minX);
  const sy = (size.height - pad * 2) / (bounds.maxY - bounds.minY);
  const scaleValue = Math.min(sx, sy);
  const centerX = size.width * 0.5;
  const centerY = size.height * 0.5;
  const baseX = pad + (point.x - bounds.minX) * scaleValue;
  const baseY = size.height - pad - (point.y - bounds.minY) * scaleValue;
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

function drawScene(state, drawing) {
  const { sceneCanvas, sceneCtx, showVoronoiOverlay } = drawing;
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
    sceneCtx.fillStyle = `hsla(${(hubIndex * 53) % 360} 70% 52% / 0.12)`;
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

export { camera, drawChart, drawScene, getSceneBounds, projectPoint, unprojectPoint };
