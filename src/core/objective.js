import { nearestCenterIndex } from "./cloud.js";
import { dijkstra, graphFromEdges, reconstructPath } from "./graph.js";
import { add, average, dist, normalize, scale, squaredDist, sub, vec } from "./math.js";

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

function evaluateObjective(hubs, edges, cloud, params) {
  const assigned = assignPoints(hubs, cloud);
  const graph = graphFromEdges(hubs, edges);
  const fx = computeFx(hubs, assigned.assignments);
  const g1Info = computePathSmoothness(hubs, assigned.weights, graph);
  const g2 = computeG2(hubs, params.lambda);
  const g3 = computeGraphVariation(hubs, graph.edges, params.mu);
  const g4 = computeStationBuildCost(hubs, params.stationCost);

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

export {
  assignPoints,
  clipPolygonWithHalfPlane,
  computeClusterEnergies,
  computeFx,
  computeG2,
  computeGraphVariation,
  computePathSmoothness,
  computeStationBuildCost,
  computeVoronoiCells,
  evaluateObjective,
  polygonCentroid,
};
