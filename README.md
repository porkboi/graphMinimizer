# graphMinimizer

Browser-based point-cloud hub optimisation demo with two solver paths:

- `ADMM` explorer and tuning flows
- `SGD` explorer and tuning flows

The codebase is now split into ES modules so geometry, solver logic, rendering, state setup, and DOM wiring can evolve independently.

## Structure

- [`app.js`](./app.js) is the thin browser entrypoint.
- [`src/app`](./src/app) contains DOM lookup and app orchestration.
- [`src/core`](./src/core) contains point-cloud generation, math helpers, graph helpers, and objective evaluation.
- [`src/solver`](./src/solver) contains the iterative ADMM and SGD update logic.
- [`src/state`](./src/state) contains state factories plus the tuning state machine.
- [`src/render`](./src/render) contains canvas projection and drawing code.
- [`docs/architecture.md`](./docs/architecture.md) documents the module boundaries.

## Local use

Serve the folder with any static file server and open `index.html`. The app uses browser-native ES modules, so loading the HTML file directly from disk may be blocked by browser module restrictions.
