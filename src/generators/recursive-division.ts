import type { Maze } from '../core/maze'
import type { Cell, GeneratorStep } from '../core/types'
import { EAST, SOUTH } from '../core/types'
import type { MazeGenerator } from './generator'

// Recursive Division works in reverse: start with NO walls (all passages open),
// then recursively add walls, leaving exactly one passage per wall segment.
export class RecursiveDivisionGenerator implements MazeGenerator {
  async *generate(maze: Maze): AsyncGenerator<GeneratorStep> {
    // Step 1: remove all internal walls so the grid is fully open
    for (let row = 0; row < maze.rows; row++) {
      for (let col = 0; col < maze.cols; col++) {
        if (row > 0)             maze.removeWall({ row, col }, { row: row - 1, col })
        if (col < maze.cols - 1) maze.removeWall({ row, col }, { row, col: col + 1 })
      }
    }

    yield { type: 'visit', cell: { row: 0, col: 0 } }

    // Step 2: recursively divide
    yield* divide(maze, 0, 0, maze.rows, maze.cols)

    yield { type: 'done', cell: { row: 0, col: 0 } }
  }
}

async function* divide(
  maze: Maze,
  rowStart: number,
  colStart: number,
  height: number,
  width: number,
): AsyncGenerator<GeneratorStep> {
  if (height < 2 || width < 2) return

  // Prefer dividing along the longer axis; break ties randomly
  const horizontal = height > width ? true : width > height ? false : Math.random() < 0.5

  if (horizontal) {
    // Wall sits between row (wallRow-1) and (wallRow)
    const wallRow = rowStart + 1 + Math.floor(Math.random() * (height - 1))
    const passCol = colStart + Math.floor(Math.random() * width)

    for (let col = colStart; col < colStart + width; col++) {
      if (col === passCol) continue
      maze.addWall({ row: wallRow - 1, col }, SOUTH)
      yield { type: 'visit', cell: { row: wallRow - 1, col } as Cell }
    }

    yield* divide(maze, rowStart, colStart, wallRow - rowStart, width)
    yield* divide(maze, wallRow, colStart, rowStart + height - wallRow, width)
  } else {
    // Wall sits between col (wallCol-1) and (wallCol)
    const wallCol = colStart + 1 + Math.floor(Math.random() * (width - 1))
    const passRow = rowStart + Math.floor(Math.random() * height)

    for (let row = rowStart; row < rowStart + height; row++) {
      if (row === passRow) continue
      maze.addWall({ row, col: wallCol - 1 }, EAST)
      yield { type: 'visit', cell: { row, col: wallCol - 1 } as Cell }
    }

    yield* divide(maze, rowStart, colStart, height, wallCol - colStart)
    yield* divide(maze, rowStart, wallCol, height, colStart + width - wallCol)
  }
}
