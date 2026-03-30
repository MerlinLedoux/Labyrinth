import { ALL_WALLS, EAST, NORTH, OPPOSITE, SOUTH, WEST, type Cell, type Direction } from './types'

export class Maze {
  readonly rows: number
  readonly cols: number
  /** walls[row][col] = bitmask of remaining walls */
  private walls: Uint8Array[]

  constructor(rows: number, cols: number) {
    this.rows = rows
    this.cols = cols
    // Every cell starts with all 4 walls
    this.walls = Array.from({ length: rows }, () => new Uint8Array(cols).fill(ALL_WALLS))
  }

  inBounds(row: number, col: number): boolean {
    return row >= 0 && row < this.rows && col >= 0 && col < this.cols
  }

  hasWall(cell: Cell, dir: Direction): boolean {
    return (this.walls[cell.row][cell.col] & dir) !== 0
  }

  /** Remove the wall between two adjacent cells */
  removeWall(a: Cell, b: Cell): void {
    const dr = b.row - a.row
    const dc = b.col - a.col
    let dir: Direction
    if (dr === -1) dir = NORTH
    else if (dr === 1) dir = SOUTH
    else if (dc === 1) dir = EAST
    else if (dc === -1) dir = WEST
    else throw new Error(`Cells are not adjacent: (${a.row},${a.col}) and (${b.row},${b.col})`)

    this.walls[a.row][a.col] &= ~dir
    this.walls[b.row][b.col] &= ~OPPOSITE[dir]
  }

  /** All valid neighbours of a cell (in bounds), regardless of walls */
  neighbors(cell: Cell): Cell[] {
    const { row, col } = cell
    const result: Cell[] = []
    if (row > 0)            result.push({ row: row - 1, col })
    if (row < this.rows - 1) result.push({ row: row + 1, col })
    if (col < this.cols - 1) result.push({ row, col: col + 1 })
    if (col > 0)            result.push({ row, col: col - 1 })
    return result
  }

  /** Neighbours reachable without crossing a wall */
  passableNeighbors(cell: Cell): Cell[] {
    const dirs: Direction[] = [NORTH, SOUTH, EAST, WEST]
    const deltas: [number, number][] = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    const result: Cell[] = []
    for (let i = 0; i < 4; i++) {
      if (!this.hasWall(cell, dirs[i])) {
        const nr = cell.row + deltas[i][0]
        const nc = cell.col + deltas[i][1]
        if (this.inBounds(nr, nc)) result.push({ row: nr, col: nc })
      }
    }
    return result
  }

  cellKey(cell: Cell): number {
    return cell.row * this.cols + cell.col
  }

  fromKey(key: number): Cell {
    return { row: Math.floor(key / this.cols), col: key % this.cols }
  }
}
