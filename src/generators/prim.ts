import type { Maze } from '../core/maze'
import type { Cell, GeneratorStep } from '../core/types'
import type { MazeGenerator } from './generator'

export class PrimGenerator implements MazeGenerator {
  async *generate(maze: Maze): AsyncGenerator<GeneratorStep> {
    const inMaze = new Set<number>()
    const frontier: Cell[] = []

    const addFrontier = (cell: Cell) => {
      const key = maze.cellKey(cell)
      if (!inMaze.has(key) && !frontier.some(f => maze.cellKey(f) === key)) {
        frontier.push(cell)
      }
    }

    const markInMaze = (cell: Cell) => {
      inMaze.add(maze.cellKey(cell))
      for (const n of maze.neighbors(cell)) {
        addFrontier(n)
      }
    }

    // Start from a random cell
    const startRow = Math.floor(Math.random() * maze.rows)
    const startCol = Math.floor(Math.random() * maze.cols)
    const start: Cell = { row: startRow, col: startCol }
    markInMaze(start)

    yield { type: 'visit', cell: start, frontier: [...frontier] }

    while (frontier.length > 0) {
      // Pick a random frontier cell
      const idx = Math.floor(Math.random() * frontier.length)
      const cell = frontier.splice(idx, 1)[0]

      // Find a neighbor that is already in the maze
      const inMazeNeighbors = maze.neighbors(cell).filter(n => inMaze.has(maze.cellKey(n)))
      if (inMazeNeighbors.length === 0) continue

      const neighbor = inMazeNeighbors[Math.floor(Math.random() * inMazeNeighbors.length)]
      maze.removeWall(cell, neighbor)
      markInMaze(cell)

      yield { type: 'visit', cell, frontier: [...frontier] }
    }

    yield { type: 'done', cell: start }
  }
}
