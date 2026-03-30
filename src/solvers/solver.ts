import type { Maze } from '../core/maze'
import type { Cell, SolverStep } from '../core/types'

export interface MazeSolver {
  solve(maze: Maze, start: Cell, end: Cell): AsyncGenerator<SolverStep>
}
