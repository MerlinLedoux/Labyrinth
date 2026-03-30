// Bitmask for walls: a cell can have walls on any combination of these sides
export const NORTH = 1
export const SOUTH = 2
export const EAST  = 4
export const WEST  = 8
export const ALL_WALLS = NORTH | SOUTH | EAST | WEST

export type Direction = typeof NORTH | typeof SOUTH | typeof EAST | typeof WEST

export const OPPOSITE: Record<number, Direction> = {
  [NORTH]: SOUTH,
  [SOUTH]: NORTH,
  [EAST]:  WEST,
  [WEST]:  EAST,
}

// A cell is identified by its (row, col) position
export interface Cell {
  row: number
  col: number
}

// States emitted by generators during animation
export type GeneratorStepType = 'visit' | 'add-frontier' | 'done'

export interface GeneratorStep {
  type: GeneratorStepType
  cell: Cell
  /** Cells currently in the frontier (Prim) */
  frontier?: Cell[]
}

// States emitted by solvers during animation
export type SolverStepType = 'open' | 'close' | 'path' | 'done'

export interface SolverStep {
  type: SolverStepType
  cell: Cell
  /** Final path cells (only when type === 'path') */
  path?: Cell[]
}
