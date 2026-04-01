import type { Maze3D, Cell3D } from '../core/maze3d'

export type Step3D =
  | { type: 'visit'; cell: Cell3D }
  | { type: 'done' }

/** Recursive backtracker (DFS) adapté au cube 3D — 6 directions. */
export async function* generateMaze3D(maze: Maze3D): AsyncGenerator<Step3D> {
  const visited = new Set<number>()
  const start: Cell3D = { row: 0, col: 0, layer: 0 }
  visited.add(maze.cellKey(start))
  const stack: Cell3D[] = [start]

  while (stack.length > 0) {
    const current = stack[stack.length - 1]
    const unvisited = maze.neighbors(current).filter(n => !visited.has(maze.cellKey(n)))

    if (unvisited.length === 0) {
      stack.pop()
    } else {
      const next = unvisited[Math.floor(Math.random() * unvisited.length)]
      maze.removeWall(current, next)
      visited.add(maze.cellKey(next))
      stack.push(next)
      yield { type: 'visit', cell: next }
    }
  }

  yield { type: 'done' }
}
