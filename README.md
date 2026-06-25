# Labyrinth

Maze generator, solver, and game — built with TypeScript, Canvas 2D, and Three.js.  
Generate mazes with 6 different algorithms, solve them with 3 solvers, explore a 3D cube maze, or play through one yourself.

> **[Try it live]()**  <!-- Replace with your deployed URL -->

<!-- Replace with a screenshot or GIF of the main 2D view -->
![Labyrinth overview]()

---

## Features

### Generate & Solve

Watch mazes being built and solved step by step. Pick an algorithm, adjust the grid size (up to 60x60), and control the animation speed.

| Algorithm | Style |
|---|---|
| Prim | Organic, many short dead-ends |
| Recursive Backtracker | Long winding corridors |
| Binary Tree | Fast, diagonal bias |
| Hunt & Kill | Iterative backtracker variant |
| Recursive Division | Geometric, straight walls |
| Kruskal | Random edge merging (Union-Find) |

Solvers: **A\***, **BFS**, **DFS**

<!-- Replace with a GIF showing maze generation -->
![Generation]()

<!-- Replace with a GIF showing maze solving -->
![Solving]()

### Cube 3D

A three-dimensional maze on a cube, rendered with Three.js. Generate, solve, and orbit around it.

<!-- Replace with a GIF of the 3D cube maze -->
![Cube 3D]()

### Play

Navigate the maze yourself with keyboard controls (arrow keys or ZQSD). Visibility is limited to 3 cells around you. You can also watch an AI agent solve it.

<!-- Replace with a GIF of the play mode -->
![Play mode]()

---

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Development server with hot reload |
| `npm run build` | Type-check + production build |
| `npm run preview` | Preview the production build |

## Project Structure

```
src/
├── core/           # Maze data structure & types
├── generators/     # 6 generation algorithms
├── solvers/        # A*, BFS, DFS
├── renderer/       # Canvas 2D & Three.js 3D renderers
└── main.ts         # Entry point, UI orchestration
```

## How It Works

Each cell stores a **bitmask of its walls** (`N=1 S=2 E=4 W=8`). Generators carve passages by removing walls between cells. Recursive Division works in reverse — it starts open and adds walls.

All algorithms are **async generators** that yield their state at each step. The UI consumes these steps with a configurable delay, driving the animation without blocking the browser.

## License

MIT
