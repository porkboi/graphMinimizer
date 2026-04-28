# Architecture

## Overview

The previous `app.js` mixed five concerns:

1. Point-cloud and graph primitives
2. Objective evaluation
3. ADMM / SGD iteration logic
4. Multi-mode state management
5. DOM and canvas rendering

Those concerns are now separated into browser-native ES modules.

## Modules

### `src/core`

- `math.js`: vector math, RNG helpers, cloning, clamping
- `cloud.js`: seeded cloud generation, hub seeding, KMeans
- `graph.js`: graph construction, shortest paths, connectivity checks
- `objective.js`: assignments, energy terms, Voronoi helpers, objective evaluation

### `src/solver`

- `optimizers.js`: ADMM proximal steps, SGD gradients, convergence loops, and edge-pruning refinement

### `src/state`

- `base.js`: app state factories and shared state helpers
- `tuning.js`: explorer stepping and the hub-tuning state machine

### `src/render`

- `canvas.js`: camera state, projection, scene drawing, chart drawing

### `src/app`

- `dom.js`: DOM references and static UI copy
- `main.js`: runtime orchestration, event binding, animation loop, and UI syncing

## Practical guidance

- Add new objective terms in `src/core/objective.js`, then surface them in `src/solver/optimizers.js` and `src/app/main.js`.
- Add new visual layers in `src/render/canvas.js` instead of mixing rendering with solver code.
- Keep DOM reads and writes inside `src/app` so solver modules stay testable and reusable.
