import type { Maze } from '../core/maze'
import type { Cell, SolverStep } from '../core/types'
import type { MazeSolver } from './solver'

export class DFSSolver implements MazeSolver {
  async *solve(maze: Maze, start: Cell, end: Cell): AsyncGenerator<SolverStep> {
    const startKey = maze.cellKey(start)
    const endKey   = maze.cellKey(end)

    const cameFrom = new Map<number, number>()
    const open     = new Set<number>([startKey])  // dans la stack, pas encore traités
    const closed   = new Set<number>()
    const stack: number[] = [startKey]

    while (stack.length > 0) {
      const currentKey = stack.pop()!
      const current    = maze.fromKey(currentKey)

      if (closed.has(currentKey)) continue

      if (currentKey === endKey) {
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
      open.delete(currentKey)
      yield { type: 'close', cell: current }

      for (const neighbor of maze.passableNeighbors(current)) {
        const nKey = maze.cellKey(neighbor)
        if (closed.has(nKey)) continue
        cameFrom.set(nKey, currentKey)
        open.add(nKey)
        stack.push(nKey)
        yield { type: 'open', cell: neighbor }
      }
    }

    yield { type: 'done', cell: start }
  }
}
