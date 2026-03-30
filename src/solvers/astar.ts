import type { Maze } from '../core/maze'
import type { Cell, SolverStep } from '../core/types'
import type { MazeSolver } from './solver'

function heuristic(a: Cell, b: Cell): number {
  return Math.abs(a.row - b.row) + Math.abs(a.col - b.col)
}

export class AStarSolver implements MazeSolver {
  async *solve(maze: Maze, start: Cell, end: Cell): AsyncGenerator<SolverStep> {
    const startKey = maze.cellKey(start)
    const endKey   = maze.cellKey(end)

    // g = cost from start, f = g + heuristic
    const g = new Map<number, number>([[startKey, 0]])
    const f = new Map<number, number>([[startKey, heuristic(start, end)]])
    const cameFrom = new Map<number, number>()
    const closed = new Set<number>()

    // Simple priority queue: array sorted by f value
    const open = new Set<number>([startKey])

    const popBest = (): number => {
      let best = -1
      let bestF = Infinity
      for (const key of open) {
        const fVal = f.get(key) ?? Infinity
        if (fVal < bestF) { bestF = fVal; best = key }
      }
      open.delete(best)
      return best
    }

    while (open.size > 0) {
      const currentKey = popBest()
      const current = maze.fromKey(currentKey)

      if (currentKey === endKey) {
        // Reconstruct path
        const path: Cell[] = []
        let k: number | undefined = endKey
        while (k !== undefined) {
          path.unshift(maze.fromKey(k))
          k = cameFrom.get(k)
        }
        yield { type: 'path', cell: current, path }
        yield { type: 'done', cell: current }
        return
      }

      closed.add(currentKey)
      yield { type: 'close', cell: current }

      for (const neighbor of maze.passableNeighbors(current)) {
        const nKey = maze.cellKey(neighbor)
        if (closed.has(nKey)) continue

        const tentativeG = (g.get(currentKey) ?? Infinity) + 1

        if (tentativeG < (g.get(nKey) ?? Infinity)) {
          cameFrom.set(nKey, currentKey)
          g.set(nKey, tentativeG)
          f.set(nKey, tentativeG + heuristic(neighbor, end))
          if (!open.has(nKey)) {
            open.add(nKey)
            yield { type: 'open', cell: neighbor }
          }
        }
      }
    }

    // No path found
    yield { type: 'done', cell: start }
  }
}
