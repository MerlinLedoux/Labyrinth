import type { Maze } from '../core/maze'
import type { Cell, GeneratorStep } from '../core/types'
import type { MazeGenerator } from './generator'

export class RecursiveBacktrackerGenerator implements MazeGenerator {
  async *generate(maze: Maze): AsyncGenerator<GeneratorStep> {
    const visited = new Set<number>() // Collection unique garder en mémoire toute les cellules déjà visiter.
    const stack: Cell[] = [] // Tableau d'array pour simuler une pile empile les cellules visiter pour que quand l'algorithme est bloquer il puise utliser cette pile pour retrouver une cellule pour continuer l'exporation.

    const startRow = Math.floor(Math.random() * maze.rows)
    const startCol = Math.floor(Math.random() * maze.cols)
    const start: Cell = { row: startRow, col: startCol }

    visited.add(maze.cellKey(start))
    stack.push(start)
    yield { type: 'visit', cell: start }

    while (stack.length > 0) {
      const current = stack[stack.length - 1]

      // Unvisited neighbours
      const unvisited = maze.neighbors(current).filter(
        n => !visited.has(maze.cellKey(n))
      )

      if (unvisited.length === 0) {
        stack.pop()
        continue
      }

      const next = unvisited[Math.floor(Math.random() * unvisited.length)]
      maze.removeWall(current, next)
      visited.add(maze.cellKey(next))
      stack.push(next)

      yield { type: 'visit', cell: next, frontier: [...stack] }
    }

    yield { type: 'done', cell: start }
  }
}
