import { dist, squaredDist } from "./math.js";

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
  const canonicalEdges = canonicalizeEdges(edges);
  for (const [a, b] of canonicalEdges) {
    const w = dist(points[a], points[b]);
    adjacency[a].push({ to: b, w });
    adjacency[b].push({ to: a, w });
  }
  return { adjacency, edges: canonicalEdges };
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

export {
  buildCompleteGraph,
  buildKnnGraph,
  buildMaxGraph,
  canonicalizeEdges,
  dijkstra,
  edgeKey,
  graphFromEdges,
  isGraphConnected,
  pathContainsEdge,
  pathSquaredLength,
  reconstructPath,
};
