import type { Maze } from '../core/maze'
import type { Cell, GeneratorStep } from '../core/types'
import type { MazeGenerator } from './generator'

// Hunt-and-Kill:
// 1. Walk randomly from current cell, carving into unvisited neighbours.
// 2. When stuck (no unvisited neighbours), "hunt": scan the grid top-left to
//    bottom-right for an unvisited cell that touches a visited one, then
//    connect them and resume the walk from there.
export class HuntAndKillGenerator implements MazeGenerator {
  async *generate(maze: Maze): AsyncGenerator<GeneratorStep> {
    const visited = new Set<number>()

    const startRow = Math.floor(Math.random() * maze.rows)
    const startCol = Math.floor(Math.random() * maze.cols)
    let current: Cell = { row: startRow, col: startCol }
    visited.add(maze.cellKey(current))

    yield { type: 'visit', cell: current }

    while (true) {
      // Walk phase: move to a random unvisited neighbour
      const unvisited = maze.neighbors(current).filter(
        n => !visited.has(maze.cellKey(n))
      )

      if (unvisited.length > 0) {
        const next = unvisited[Math.floor(Math.random() * unvisited.length)]
        maze.removeWall(current, next)
        visited.add(maze.cellKey(next))
        current = next
        yield { type: 'visit', cell: current }
      } else {
        // Hunt phase: scan for an unvisited cell adjacent to a visited one
        let found = false
        hunt:
        for (let row = 0; row < maze.rows; row++) {
          for (let col = 0; col < maze.cols; col++) {
            const cell: Cell = { row, col }
            if (visited.has(maze.cellKey(cell))) continue

            const visitedNeighbors = maze.neighbors(cell).filter(
              n => visited.has(maze.cellKey(n))
            )
            if (visitedNeighbors.length > 0) {
              const neighbor = visitedNeighbors[Math.floor(Math.random() * visitedNeighbors.length)]
              maze.removeWall(cell, neighbor)
              visited.add(maze.cellKey(cell))
              current = cell
              yield { type: 'visit', cell: current }
              found = true
              break hunt
            }
          }
        }
        if (!found) break // All cells visited
      }
    }

    yield { type: 'done', cell: current }
  }
}
