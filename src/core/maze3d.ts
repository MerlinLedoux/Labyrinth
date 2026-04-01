// Directions bitmask (6 faces du cube)
export const NORTH3 =  1
export const SOUTH3 =  2
export const EAST3  =  4
export const WEST3  =  8
export const UP3    = 16
export const DOWN3  = 32

const ALL_WALLS_3D = 63  // 1|2|4|8|16|32

const OPPOSITE_3D: Record<number, number> = {
  [NORTH3]: SOUTH3, [SOUTH3]: NORTH3,
  [EAST3]:  WEST3,  [WEST3]:  EAST3,
  [UP3]:    DOWN3,  [DOWN3]:  UP3,
}

export const DIRS_3D = [NORTH3, SOUTH3, EAST3, WEST3, UP3, DOWN3] as const

// [drow, dcol, dlayer]
const DELTA: Record<number, [number, number, number]> = {
  [NORTH3]: [-1,  0,  0],
  [SOUTH3]: [ 1,  0,  0],
  [EAST3]:  [ 0,  1,  0],
  [WEST3]:  [ 0, -1,  0],
  [UP3]:    [ 0,  0,  1],
  [DOWN3]:  [ 0,  0, -1],
}

export type Cell3D = { row: number; col: number; layer: number }

export class Maze3D {
  readonly rows:   number
  readonly cols:   number
  readonly layers: number
  /** walls[layer][row][col] = bitmask des murs restants */
  private walls: Uint8Array[][]

  constructor(rows: number, cols: number, layers: number) {
    this.rows   = rows
    this.cols   = cols
    this.layers = layers
    this.walls  = Array.from({ length: layers }, () =>
      Array.from({ length: rows }, () => new Uint8Array(cols).fill(ALL_WALLS_3D))
    )
  }

  inBounds(row: number, col: number, layer: number): boolean {
    return row >= 0 && row < this.rows
        && col >= 0 && col < this.cols
        && layer >= 0 && layer < this.layers
  }

  hasWall(cell: Cell3D, dir: number): boolean {
    return (this.walls[cell.layer][cell.row][cell.col] & dir) !== 0
  }

  removeWall(a: Cell3D, b: Cell3D): void {
    const dr = b.row - a.row
    const dc = b.col - a.col
    const dl = b.layer - a.layer
    const dir =
      dr === -1 ? NORTH3 : dr === 1 ? SOUTH3 :
      dc ===  1 ? EAST3  : dc === -1 ? WEST3 :
      dl ===  1 ? UP3    : DOWN3
    this.walls[a.layer][a.row][a.col] &= ~dir
    this.walls[b.layer][b.row][b.col] &= ~OPPOSITE_3D[dir]
  }

  neighbors(cell: Cell3D): Cell3D[] {
    const result: Cell3D[] = []
    for (const dir of DIRS_3D) {
      const [dr, dc, dl] = DELTA[dir]
      const nr = cell.row + dr, nc = cell.col + dc, nl = cell.layer + dl
      if (this.inBounds(nr, nc, nl)) result.push({ row: nr, col: nc, layer: nl })
    }
    return result
  }

  passableNeighbors(cell: Cell3D): Cell3D[] {
    const result: Cell3D[] = []
    for (const dir of DIRS_3D) {
      if (this.hasWall(cell, dir)) continue
      const [dr, dc, dl] = DELTA[dir]
      const nr = cell.row + dr, nc = cell.col + dc, nl = cell.layer + dl
      if (this.inBounds(nr, nc, nl)) result.push({ row: nr, col: nc, layer: nl })
    }
    return result
  }

  cellKey(cell: Cell3D): number {
    return cell.layer * this.rows * this.cols + cell.row * this.cols + cell.col
  }

  fromKey(key: number): Cell3D {
    const rc = this.rows * this.cols
    return {
      layer: Math.floor(key / rc),
      row:   Math.floor((key % rc) / this.cols),
      col:   key % this.cols,
    }
  }
}
