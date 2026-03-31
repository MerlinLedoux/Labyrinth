import { Maze } from './core/maze'
import type { Cell } from './core/types'
import type { MazeGenerator } from './generators/generator'
import { PrimGenerator } from './generators/prim'
import { RecursiveBacktrackerGenerator } from './generators/recursive-backtracker'
import { KruskalGenerator } from './generators/kruskal'
import { BinaryTreeGenerator } from './generators/binary-tree'
import { HuntAndKillGenerator } from './generators/hunt-and-kill'
import { RecursiveDivisionGenerator } from './generators/recursive-division'
import { AStarSolver } from './solvers/astar'
import { Renderer2D } from './renderer/renderer2d'

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas        = document.getElementById('maze-canvas') as HTMLCanvasElement
const btnGen        = document.getElementById('btn-generate') as HTMLButtonElement
const btnSolve      = document.getElementById('btn-solve') as HTMLButtonElement
const sizeSlider    = document.getElementById('size-slider') as HTMLInputElement
const sizeLabel     = document.getElementById('size-label') as HTMLSpanElement
const speedSlider   = document.getElementById('speed-slider') as HTMLInputElement
const speedLabel    = document.getElementById('speed-label') as HTMLSpanElement
const genSelect     = document.getElementById('gen-select') as HTMLSelectElement

// ── Generator registry ────────────────────────────────────────────────────────
const GENERATORS: Record<string, MazeGenerator> = {
  'binary-tree':          new BinaryTreeGenerator(),
  'hunt-and-kill':        new HuntAndKillGenerator(),
  'prim':                 new PrimGenerator(),
  'recursive-backtracker': new RecursiveBacktrackerGenerator(),
  'recursive-division':   new RecursiveDivisionGenerator(),
  'kruskal':              new KruskalGenerator(), 
}

// ── State ─────────────────────────────────────────────────────────────────────
let maze: Maze | null = null
let running = false

const renderer = new Renderer2D(canvas)
const solver   = new AStarSolver()

// ── Helpers ───────────────────────────────────────────────────────────────────
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function getSpeed(): number {
  return Number(speedSlider.value)
}

function resizeCanvas(): void {
  const size = Math.min(window.innerWidth, window.innerHeight) * 0.75
  canvas.width  = size
  canvas.height = size
}

function setControls(disabled: boolean): void {
  btnGen.disabled    = disabled
  btnSolve.disabled  = disabled
  genSelect.disabled = disabled
}

// ── Generate ──────────────────────────────────────────────────────────────────
async function generate(): Promise<void> {
  if (running) return
  running = true
  setControls(true)

  const gridSize = Number(sizeSlider.value)
  maze = new Maze(gridSize, gridSize)

  const inMaze   = new Set<number>()
  const frontier = new Set<number>()
  const start: Cell = { row: 0, col: 0 }
  const end: Cell   = { row: gridSize - 1, col: gridSize - 1 }

  const generator = GENERATORS[genSelect.value] ?? GENERATORS['prim']

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
  setControls(false)
}

// ── Solve ─────────────────────────────────────────────────────────────────────
async function solve(): Promise<void> {
  if (!maze || running) return
  running = true
  setControls(true)

  const currentMaze = maze
  const gridSize = currentMaze.rows
  const start: Cell = { row: 0, col: 0 }
  const end: Cell   = { row: gridSize - 1, col: gridSize - 1 }

  const inMaze = new Set<number>()
  for (let r = 0; r < currentMaze.rows; r++)
    for (let c = 0; c < currentMaze.cols; c++)
      inMaze.add(currentMaze.cellKey({ row: r, col: c }))

  const open   = new Set<number>()
  const closed = new Set<number>()
  let   path: Cell[] = []

  for await (const step of solver.solve(currentMaze, start, end)) {
    if (step.type === 'open')  open.add(currentMaze.cellKey(step.cell))
    if (step.type === 'close') { closed.add(currentMaze.cellKey(step.cell)); open.delete(currentMaze.cellKey(step.cell)) }
    if (step.type === 'path')  path = step.path ?? []
    if (step.type === 'done')  break
    renderer.drawMaze(currentMaze, { inMaze, open, closed, path, start, end })
    await delay(getSpeed())
  }

  renderer.drawMaze(currentMaze, { inMaze, open, closed, path, start, end })

  running = false
  setControls(false)
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
