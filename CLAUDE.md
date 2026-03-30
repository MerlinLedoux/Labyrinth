# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev      # Start dev server (http://localhost:5173)
npm run build    # Type-check + production build
npm run preview  # Preview production build locally
```

## Architecture

**Stack:** TypeScript + Vite, Canvas 2D API (no framework, no external runtime deps).

### Data structure
`Maze` ([src/core/maze.ts](src/core/maze.ts)) stores a grid of cells where each cell's walls are a bitmask (`N=1, S=2, E=4, W=8`). Key methods: `removeWall(a, b)`, `hasWall(cell, dir)`, `neighbors(cell)`, `passableNeighbors(cell)`.

### Generators & Solvers — Async Generators
All generators and solvers are `async function*` that **yield** state at each algorithm step. This decouples the algorithm from the animation: the runner in [src/main.ts](src/main.ts) calls `await delay(ms)` between each `next()` call.

- Generator interface: [src/generators/generator.ts](src/generators/generator.ts) — yields `GeneratorStep`
- Solver interface: [src/solvers/solver.ts](src/solvers/solver.ts) — yields `SolverStep`
- Current generator: Prim ([src/generators/prim.ts](src/generators/prim.ts))
- Current solver: A* ([src/solvers/astar.ts](src/solvers/astar.ts))

To add a new algorithm: implement the interface, import it in [src/main.ts](src/main.ts), wire it to a UI button.

### Renderer
[src/renderer/renderer2d.ts](src/renderer/renderer2d.ts) — single `drawMaze(maze, options)` call redraws the full canvas. `options` accepts sets of cell keys for each visual state (`inMaze`, `frontier`, `open`, `closed`, `path`).

[src/renderer/renderer3d.ts](src/renderer/renderer3d.ts) — placeholder for future Three.js 3D renderer. The same `Maze` data structure and generator/solver interfaces will be reused.

### Entry point
[src/main.ts](src/main.ts) orchestrates everything: reads UI controls, runs the generator/solver loops, and calls the renderer after each step.

## Extending the project

- **New generator:** implement `MazeGenerator` → add a button/select in `index.html` → wire in `main.ts`
- **New solver:** implement `MazeSolver` → same pattern
- **3D view:** implement `Renderer3D` using Three.js, feed it the same `Maze` object
