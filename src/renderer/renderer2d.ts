import type { Maze } from '../core/maze'
import { EAST, NORTH, SOUTH, WEST } from '../core/types'
import type { Cell } from '../core/types'

const COLORS = {
  background:  '#1a1a2e',
  wall:        '#e0e0e0',
  unvisited:   '#1a1a2e',
  inMaze:      '#16213e',
  frontier:    '#0f3460',
  open:        '#ffd600',   // A* open set — green
  closed:      '#e53935',   // A* closed set — red
  path:        '#4caf50',   // Final path — yellow
  start:       '#29b6f6',   // Start cell — light blue
  end:         '#ab47bc',   // End cell — purple
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

    ctx.fillStyle = COLORS.background
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)

    for (let row = 0; row < maze.rows; row++) {
      for (let col = 0; col < maze.cols; col++) {
        const x = col * cs
        const y = row * cs
        const key = maze.cellKey({ row, col })

        // Cell fill
        let fill = COLORS.unvisited
        if (options.path?.some(c => c.row === row && c.col === col)) fill = COLORS.path
        else if (options.closed?.has(key)) fill = COLORS.closed
        else if (options.open?.has(key))   fill = COLORS.open
        else if (options.frontier?.has(key)) fill = COLORS.frontier
        else if (options.inMaze?.has(key)) fill = COLORS.inMaze

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

    // Start / End markers
    if (options.start) this.drawMarker(options.start, COLORS.start)
    if (options.end)   this.drawMarker(options.end,   COLORS.end)
  }

  private drawMarker(cell: Cell, color: string): void {
    const cs = this.cellSize
    const x = cell.col * cs + cs / 2
    const y = cell.row * cs + cs / 2
    const r = cs * 0.3
    this.ctx.beginPath()
    this.ctx.arc(x, y, r, 0, Math.PI * 2)
    this.ctx.fillStyle = color
    this.ctx.fill()
  }
}
