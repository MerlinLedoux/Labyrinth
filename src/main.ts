import { Maze } from './core/maze'
import type { Cell } from './core/types'
import { PrimGenerator } from './generators/prim'
import { AStarSolver } from './solvers/astar'
import { Renderer2D } from './renderer/renderer2d'

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas     = document.getElementById('maze-canvas') as HTMLCanvasElement
const btnGen     = document.getElementById('btn-generate') as HTMLButtonElement
const btnSolve   = document.getElementById('btn-solve') as HTMLButtonElement
const sizeSlider = document.getElementById('size-slider') as HTMLInputElement
const sizeLabel  = document.getElementById('size-label') as HTMLSpanElement
const speedSlider= document.getElementById('speed-slider') as HTMLInputElement
const speedLabel = document.getElementById('speed-label') as HTMLSpanElement

// ── State ─────────────────────────────────────────────────────────────────────
let maze: Maze | null = null
let running = false

const renderer = new Renderer2D(canvas)
const generator = new PrimGenerator()
const solver    = new AStarSolver()

// ── Helpers ───────────────────────────────────────────────────────────────────
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function getSpeed(): number {
  // slider: 0 (fast) → 100 (slow). Map to ms: 0 → 0ms, 100 → 100ms
  return Number(speedSlider.value)
}

function resizeCanvas(): void {
  const size = Math.min(window.innerWidth, window.innerHeight) * 0.75
  canvas.width  = size
  canvas.height = size
}

// ── Generate ──────────────────────────────────────────────────────────────────
async function generate(): Promise<void> {
  if (running) return
  running = true
  btnGen.disabled   = true
  btnSolve.disabled = true

  const gridSize = Number(sizeSlider.value)
  maze = new Maze(gridSize, gridSize)

  const inMaze   = new Set<number>()
  const frontier = new Set<number>()
  const start: Cell = { row: 0, col: 0 }
  const end: Cell   = { row: gridSize - 1, col: gridSize - 1 }

  for await (const step of generator.generate(maze)) {
    if (step.type === 'visit') {
      inMaze.add(maze.cellKey(step.cell))
      frontier.delete(maze.cellKey(step.cell))
      step.frontier?.forEach(f => frontier.add(maze.cellKey(f)))
      renderer.drawMaze(maze, { inMaze, frontier, start, end })
    }
    if (step.type === 'done') break
    await delay(getSpeed())
  }

  renderer.drawMaze(maze, { inMaze, start, end })

  running = false
  btnGen.disabled   = false
  btnSolve.disabled = false
}

// ── Solve ─────────────────────────────────────────────────────────────────────
async function solve(): Promise<void> {
  if (!maze || running) return
  running = true
  btnGen.disabled   = true
  btnSolve.disabled = true

  const gridSize = maze.rows
  const start: Cell = { row: 0, col: 0 }
  const end: Cell   = { row: gridSize - 1, col: gridSize - 1 }

  const inMaze = new Set<number>()
  for (let r = 0; r < maze.rows; r++)
    for (let c = 0; c < maze.cols; c++)
      inMaze.add(maze.cellKey({ row: r, col: c }))

  const open   = new Set<number>()
  const closed  = new Set<number>()
  let   path: Cell[] = []

  for await (const step of solver.solve(maze, start, end)) {
    if (step.type === 'open')   open.add(maze.cellKey(step.cell))
    if (step.type === 'close') { closed.add(maze.cellKey(step.cell)); open.delete(maze.cellKey(step.cell)) }
    if (step.type === 'path')   path = step.path ?? []
    if (step.type === 'done')   break
    renderer.drawMaze(maze, { inMaze, open, closed, path, start, end })
    await delay(getSpeed())
  }

  renderer.drawMaze(maze, { inMaze, open, closed, path, start, end })

  running = false
  btnGen.disabled   = false
  btnSolve.disabled = false
}

// ── UI events ─────────────────────────────────────────────────────────────────
sizeSlider.addEventListener('input', () => {
  sizeLabel.textContent = `${sizeSlider.value}×${sizeSlider.value}`
})

speedSlider.addEventListener('input', () => {
  speedLabel.textContent = `${speedSlider.value} ms`
})

btnGen.addEventListener('click', generate)
btnSolve.addEventListener('click', solve)

window.addEventListener('resize', () => {
  resizeCanvas()
  if (maze) renderer.drawMaze(maze, {})
})

// ── Init ──────────────────────────────────────────────────────────────────────
resizeCanvas()
generate()
