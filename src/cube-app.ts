import { Maze3D } from './core/maze3d'
import { generateMaze3D } from './generators3d/backtracker3d'
import { solveMaze3D } from './solvers3d/bfs3d'
import { RendererCube } from './renderer/renderer-cube'
import type { Cell3D } from './core/maze3d'

const SIZE = 10   // 10×10×10

// ── DOM ───────────────────────────────────────────────────────────────────────
const canvas     = document.getElementById('canvas-cube')      as HTMLCanvasElement
const btnGen     = document.getElementById('btn-cube-gen')     as HTMLButtonElement
const btnSolve   = document.getElementById('btn-cube-solve')   as HTMLButtonElement
const speedInput = document.getElementById('cube-speed')       as HTMLInputElement
const speedLabel = document.getElementById('cube-speed-label') as HTMLSpanElement

// ── State ─────────────────────────────────────────────────────────────────────
let maze3d  : Maze3D        | null = null
let renderer: RendererCube  | null = null
let running                        = false

const rs = {
  inMaze  : new Set<number>(),
  open    : new Set<number>(),
  closed  : new Set<number>(),
  path    : [] as Cell3D[],
  start   : { row: 0,      col: 0,      layer: 0      } as Cell3D,
  end     : { row: SIZE-1, col: SIZE-1, layer: SIZE-1 } as Cell3D,
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function delay(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)) }
function getSpeed(): number { return Number(speedInput.value) }

function setControls(disabled: boolean): void {
  btnGen.disabled   = disabled
  btnSolve.disabled = disabled
}

function draw(): void {
  if (!maze3d || !renderer) return
  renderer.drawMaze(maze3d, rs)
}

// ── Generate ──────────────────────────────────────────────────────────────────
async function generate(): Promise<void> {
  if (running) return
  running = true
  setControls(true)

  maze3d = new Maze3D(SIZE, SIZE, SIZE)
  rs.inMaze.clear()
  rs.open.clear()
  rs.closed.clear()
  rs.path = []

  const m = maze3d

  for await (const step of generateMaze3D(m)) {
    if (step.type === 'visit') {
      rs.inMaze.add(m.cellKey(step.cell))
      draw()
    }
    if (step.type === 'done') break
    await delay(getSpeed())
  }

  draw()
  running = false
  setControls(false)
}

// ── Solve ─────────────────────────────────────────────────────────────────────
async function solve(): Promise<void> {
  if (!maze3d || running) return
  running = true
  setControls(true)

  rs.open.clear()
  rs.closed.clear()
  rs.path = []

  const m = maze3d

  for await (const step of solveMaze3D(m, rs.start, rs.end)) {
    if (step.type === 'open') {
      rs.open.add(m.cellKey(step.cell))
      draw()
    } else if (step.type === 'close') {
      rs.closed.add(m.cellKey(step.cell))
      rs.open.delete(m.cellKey(step.cell))
      draw()
    } else if (step.type === 'done') {
      rs.path = step.path
      draw()
      break
    }
    await delay(getSpeed())
  }

  running = false
  setControls(false)
}

// ── Init ──────────────────────────────────────────────────────────────────────
export function initCubeApp(): void {
  renderer = new RendererCube(canvas)
  btnGen.addEventListener('click',   generate)
  btnSolve.addEventListener('click', solve)
  speedInput.addEventListener('input', () => {
    speedLabel.textContent = `${speedInput.value} ms`
  })
  generate()
}

export function resizeCubeCanvas(): void {
  const controls = document.getElementById('cube-controls')!
  const padding  = 80
  const size = Math.min(
    window.innerWidth  - 40,
    window.innerHeight - controls.offsetHeight - padding,
  )
  canvas.width  = Math.max(300, size)
  canvas.height = Math.max(300, size)
  renderer?.resize()
}
