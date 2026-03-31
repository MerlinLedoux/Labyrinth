// Bitmask for walls: a cell can have walls on any combination of these sides
export const NORTH = 1
export const SOUTH = 2
export const EAST  = 4
export const WEST  = 8
export const ALL_WALLS = NORTH | SOUTH | EAST | WEST


// typeof extract the exact type of a variable. 
// With type SOUTH is export as 2 and not as number so if later in the code someone tries to add a wall as 3 it will return an error because 3 is not in ALL_WALLS 
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
export type GeneratorStepType = 'visit' | 'done'

// ? mean optianal
export interface GeneratorStep {
  type: GeneratorStepType
  cell: Cell
  /** Cells currently in the frontier (Prim) */
  frontier?: Cell[]
}

// States emitted by solvers during animation
// 4 state inside A* solver : open (discover a new cell add to waiting list), close (the cell have been visited and wont ever be again), path (The finish line was reach now the path back must be fin), done (the shortest path between start and finish was find).
export type SolverStepType = 'open' | 'close' | 'path' | 'done'

export interface SolverStep {
  type: SolverStepType
  cell: Cell
  /** Final path cells (only when type === 'path') */
  path?: Cell[]
}
