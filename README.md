# Labyrinth

A browser-based maze generator and solver built with TypeScript and Vite. Watch mazes being built and solved step by step with real-time animations.

## Features

- **6 generation algorithms** — each produces a visually distinct maze style
- **3 solvers** — A*, BFS, and DFS with animated exploration
- **Fully animated** — watch every step of generation and solving with adjustable speed
- **Configurable grid size** — from 5×5 to 60×60

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Generation Algorithms

| Algorithm | Maze style |
|---|---|
| **Prim** | Organic, many short dead-ends |
| **Recursive Backtracker** | Long winding corridors, few branches |
| **Binary Tree** | Extremely fast; noticeable north-east diagonal bias |
| **Hunt & Kill** | Similar to Backtracker but fully iterative |
| **Recursive Division** | Geometric rooms separated by straight walls |
| **Kruskal** | Random edge merging via Union-Find |

## How It Works

The maze is stored as a grid where each cell holds a **bitmask of its remaining walls** (`N=1 S=2 E=4 W=8`). Algorithms carve passages by removing walls between adjacent cells (Recursive Division does the opposite — it starts open and adds walls).

Every algorithm is an **async generator** (`async function*`) that yields its state after each step. The UI runner consumes these steps with a configurable delay, which is what drives the animation without blocking the browser thread.

## Project Structure

```
src/
├── core/
│   ├── types.ts              # Shared types: Cell, Direction, step states
│   └── maze.ts               # Maze class — grid + wall bitmask operations
├── generators/
│   ├── generator.ts          # MazeGenerator interface
│   ├── prim.ts
│   ├── recursive-backtracker.ts
│   ├── binary-tree.ts
│   ├── hunt-and-kill.ts
│   ├── recursive-division.ts
│   └── kruskal.ts
├── solvers/
│   ├── solver.ts             # MazeSolver interface
│   ├── astar.ts
│   ├── bfs.ts
│   └── dfs.ts
└── renderer/
    ├── renderer2d.ts         # Canvas 2D rendering
    └── renderer3d.ts         # Placeholder for future Three.js 3D view
```

## Roadmap

- [ ] Additional solvers (Wall Follower, Dijkstra)
- [ ] More generation algorithms (Wilson, Eller)
- [ ] 3D view with Three.js
- [ ] Export maze as image or JSON

## Scripts

```bash
npm run dev      # Development server with hot reload
npm run build    # Type-check + production build
npm run preview  # Preview the production build locally
```
