import type { Maze } from '../core/maze'
import { EAST, NORTH, SOUTH, WEST } from '../core/types'
import type { Cell } from '../core/types'

const COLORS = {
  background:  '#C7C7C7',
  wall:        '#1a1a1a',
  unvisited:   '#7D7D7D',
  inMaze:      '#B8B8B8',
  frontier:    '#90caf9',
  open:        '#9FCF65',
  closed:      '#65ADCF',
  path:        '#C94747',
  start:       '#81d4fa',
  end:         '#9E3C3C',
}

export class Renderer2D {
  private ctx: CanvasRenderingContext2D
  private cellSize = 0

  constructor(private canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Could not get 2D canvas context')
    this.ctx = ctx
  }

  private computeCellSize(maze: Maze): number {
    return Math.floor(Math.min(
      this.canvas.width  / maze.cols,
      this.canvas.height / maze.rows,
    ))
  }

  /** Full re-render from scratch */
  drawMaze(
    maze: Maze,
    options: {
      inMaze?: Set<number>
      frontier?: Set<number>
      open?: Set<number>
      closed?: Set<number>
      path?: Cell[]
      start?: Cell
      end?: Cell
    } = {},
  ): void {
    const { ctx } = this
    this.cellSize = this.computeCellSize(maze)
    const cs = this.cellSize

    // Center the maze in the canvas
    const ox = Math.floor((this.canvas.width  - maze.cols * cs) / 2)
    const oy = Math.floor((this.canvas.height - maze.rows * cs) / 2)

    ctx.fillStyle = COLORS.background
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)

    for (let row = 0; row < maze.rows; row++) {
      for (let col = 0; col < maze.cols; col++) {
        const x = ox + col * cs
        const y = oy + row * cs
        const key = maze.cellKey({ row, col })

        // Cell fill
        const pathFound = (options.path?.length ?? 0) > 0
        let fill = COLORS.unvisited
        if (options.path?.some(c => c.row === row && c.col === col)) fill = COLORS.path
        else if (options.closed?.has(key))   fill = pathFound ? 'rgba(101,173,207,0.3)' : COLORS.closed
        else if (options.open?.has(key))     fill = pathFound ? 'rgba(159,207,101,0.3)' : COLORS.open
        else if (options.frontier?.has(key)) fill = COLORS.frontier
        else if (options.inMaze?.has(key))   fill = COLORS.inMaze

        ctx.fillStyle = fill
        ctx.fillRect(x + 1, y + 1, cs - 1, cs - 1)

        // Walls
        ctx.strokeStyle = COLORS.wall
        ctx.lineWidth = 1.5
        if (maze.hasWall({ row, col }, NORTH)) {
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + cs, y); ctx.stroke()
        }
        if (maze.hasWall({ row, col }, SOUTH)) {
          ctx.beginPath(); ctx.moveTo(x, y + cs); ctx.lineTo(x + cs, y + cs); ctx.stroke()
        }
        if (maze.hasWall({ row, col }, WEST)) {
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + cs); ctx.stroke()
        }
        if (maze.hasWall({ row, col }, EAST)) {
          ctx.beginPath(); ctx.moveTo(x + cs, y); ctx.lineTo(x + cs, y + cs); ctx.stroke()
        }
      }
    }

    // Outer border
    ctx.strokeStyle = COLORS.wall
    ctx.lineWidth = 1.5
    ctx.strokeRect(ox, oy, maze.cols * cs, maze.rows * cs)

    // Start / End markers
    if (options.start) this.drawMarker(options.start, COLORS.start, ox, oy)
    if (options.end)   this.drawMarker(options.end,   COLORS.end,   ox, oy)
  }

  private drawMarker(cell: Cell, color: string, ox: number, oy: number): void {
    const cs = this.cellSize
    const x = ox + cell.col * cs + cs / 2
    const y = oy + cell.row * cs + cs / 2
    const r = cs * 0.3
    this.ctx.beginPath()
    this.ctx.arc(x, y, r, 0, Math.PI * 2)
    this.ctx.fillStyle = color
    this.ctx.fill()
  }
}
