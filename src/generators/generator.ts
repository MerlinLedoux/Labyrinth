import type { Maze } from '../core/maze'
import type { GeneratorStep } from '../core/types'

export interface MazeGenerator {
  generate(maze: Maze): AsyncGenerator<GeneratorStep>
}
