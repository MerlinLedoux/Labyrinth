import type { Maze3D, Cell3D } from '../core/maze3d'

export type SolverStep3D =
  | { type: 'open';  cell: Cell3D }
  | { type: 'close'; cell: Cell3D }
  | { type: 'done';  path: Cell3D[] }

/** BFS — chemin optimal dans le labyrinthe 3D. */
export async function* solveMaze3D(
  maze: Maze3D,
  start: Cell3D,
  end: Cell3D,
): AsyncGenerator<SolverStep3D> {
  const startKey = maze.cellKey(start)
  const endKey   = maze.cellKey(end)
  const visited  = new Set<number>([startKey])
  const parent   = new Map<number, number>()
  const queue: Cell3D[] = [start]

  while (queue.length > 0) {
    const current    = queue.shift()!
    const currentKey = maze.cellKey(current)

    yield { type: 'close', cell: current }

    if (currentKey === endKey) {
      // Reconstruit le chemin en remontant les parents
      const path: Cell3D[] = []
      let k: number | undefined = currentKey
      while (k !== undefined) {
        path.unshift(maze.fromKey(k))
        k = parent.get(k)
      }
      yield { type: 'done', path }
      return
    }

    for (const neighbor of maze.passableNeighbors(current)) {
      const nk = maze.cellKey(neighbor)
      if (!visited.has(nk)) {
        visited.add(nk)
        parent.set(nk, currentKey)
        queue.push(neighbor)
        yield { type: 'open', cell: neighbor }
      }
    }
  }

  yield { type: 'done', path: [] }
}
