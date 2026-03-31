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
`Maze` ([src/core/maze.ts](src/core/maze.ts)) stores a grid of cells where each cell's walls are a bitmask (`N=1, S=2, E=4, W=8`). Key methods: `removeWall(a, b)`, `addWall(cell, dir)`, `hasWall(cell, dir)`, `neighbors(cell)`, `passableNeighbors(cell)`.

`addWall` is needed by Recursive Division (the only algorithm that adds walls instead of carving passages). All other generators only call `removeWall`.

### Generators & Solvers — Async Generators
All generators and solvers are `async function*` that **yield** state at each algorithm step. This decouples the algorithm from the animation: the runner in [src/main.ts](src/main.ts) calls `await delay(ms)` between each `next()` call.

- Generator interface: [src/generators/generator.ts](src/generators/generator.ts) — yields `GeneratorStep`
- Solver interface: [src/solvers/solver.ts](src/solvers/solver.ts) — yields `SolverStep`

**Generators available:**

| Key | File | Style |
|---|---|---|
| `prim` | [src/generators/prim.ts](src/generators/prim.ts) | Organic, many short dead-ends |
| `recursive-backtracker` | [src/generators/recursive-backtracker.ts](src/generators/recursive-backtracker.ts) | Long winding corridors |
| `binary-tree` | [src/generators/binary-tree.ts](src/generators/binary-tree.ts) | Fast, strong NE diagonal bias |
| `hunt-and-kill` | [src/generators/hunt-and-kill.ts](src/generators/hunt-and-kill.ts) | Similar to backtracker, iterative |
| `recursive-division` | [src/generators/recursive-division.ts](src/generators/recursive-division.ts) | Geometric, long straight corridors |

**Solvers available:**

| Key | File | |
|---|---|---|
| A* | [src/solvers/astar.ts](src/solvers/astar.ts) | Optimal shortest path |

### Generator registry
Generators are registered in the `GENERATORS` map in [src/main.ts](src/main.ts). The UI `<select>` order is controlled by the `<option>` order in [index.html](index.html). The default generator on page load is the first `<option>`.

To add a new generator: implement `MazeGenerator` → add it to `GENERATORS` in `main.ts` → add an `<option>` in `index.html`.

### Renderer
[src/renderer/renderer2d.ts](src/renderer/renderer2d.ts) — single `drawMaze(maze, options)` call redraws the full canvas. `options` accepts sets of cell keys for each visual state (`inMaze`, `frontier`, `open`, `closed`, `path`).

[src/renderer/renderer3d.ts](src/renderer/renderer3d.ts) — placeholder for future Three.js 3D renderer. The same `Maze` data structure and generator/solver interfaces will be reused.

### Entry point
[src/main.ts](src/main.ts) orchestrates everything: reads UI controls (size, speed, generator select), runs the generator/solver `for await` loops, and calls the renderer after each step. Controls are disabled while an animation is running.
