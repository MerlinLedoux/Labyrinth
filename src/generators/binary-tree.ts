import type { Maze } from '../core/maze'
import type { Cell, GeneratorStep } from '../core/types'
import type { MazeGenerator } from './generator'

// For each cell, randomly carve either North or East (if available).
// Produces a strong diagonal bias but is extremely fast and simple.
export class BinaryTreeGenerator implements MazeGenerator {
  async *generate(maze: Maze): AsyncGenerator<GeneratorStep> {
    for (let row = 0; row < maze.rows; row++) {
      for (let col = 0; col < maze.cols; col++) {
        const cell: Cell = { row, col }
        const candidates: Cell[] = []

        if (row > 0) candidates.push({ row: row - 1, col })   // North
        if (col < maze.cols - 1) candidates.push({ row, col: col + 1 }) // East

        if (candidates.length > 0) {
          const neighbor = candidates[Math.floor(Math.random() * candidates.length)]
          maze.removeWall(cell, neighbor)
        }

        yield { type: 'visit', cell }
      }
    }

    yield { type: 'done', cell: { row: 0, col: 0 } }
  }
}
