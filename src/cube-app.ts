import { Maze3D } from './core/maze3d'
import { EAST3, SOUTH3, UP3 } from './core/maze3d'
import { generateMaze3D } from './generators3d/backtracker3d'
import { solveMaze3D } from './solvers3d/bfs3d'
import { RendererCube } from './renderer/renderer-cube'
import type { Cell3D } from './core/maze3d'

// ── DOM ───────────────────────────────────────────────────────────────────────
const canvas          = document.getElementById('canvas-cube')         as HTMLCanvasElement
const btnGen          = document.getElementById('btn-cube-gen')        as HTMLButtonElement
const btnSolve        = document.getElementById('btn-cube-solve')      as HTMLButtonElement
const speedInput      = document.getElementById('cube-speed')          as HTMLInputElement
const speedLabel      = document.getElementById('cube-speed-label')    as HTMLSpanElement
const sizeInput       = document.getElementById('cube-size')           as HTMLInputElement
const sizeLabel       = document.getElementById('cube-size-label')     as HTMLSpanElement
const openingsInput   = document.getElementById('cube-openings')       as HTMLInputElement
const openingsLabel   = document.getElementById('cube-openings-label') as HTMLSpanElement
const cubeTitle       = document.getElementById('cube-title')          as HTMLElement

// ── State ─────────────────────────────────────────────────────────────────────
let maze3d  : Maze3D        | null = null
let renderer: RendererCube  | null = null
let running                        = false

function getSize(): number { return Number(sizeInput.value) }

const rs = {
  inMaze : new Set<number>(),
  open   : new Set<number>(),
  closed : new Set<number>(),
  path   : [] as Cell3D[],
  start  : { row: 0, col: 0, layer: 0 } as Cell3D,
  end    : { row: 0, col: 0, layer: 0 } as Cell3D,
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

  const size = getSize()
  rs.start = { row: 0,      col: 0,      layer: 0      }
  rs.end   = { row: size-1, col: size-1, layer: size-1 }

  maze3d = new Maze3D(size, size, size)
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

  // ── Extra openings ────────────────────────────────────────────────────────
  const extra = Number(openingsInput.value)
  if (extra > 0) {
    const walls: [Cell3D, Cell3D][] = []
    for (let l = 0; l < size; l++)
      for (let r = 0; r < size; r++)
        for (let c = 0; c < size; c++) {
          const cell: Cell3D = { row: r, col: c, layer: l }
          if (c + 1 < size && m.hasWall(cell, EAST3))
            walls.push([cell, { row: r, col: c + 1, layer: l }])
          if (r + 1 < size && m.hasWall(cell, SOUTH3))
            walls.push([cell, { row: r + 1, col: c, layer: l }])
          if (l + 1 < size && m.hasWall(cell, UP3))
            walls.push([cell, { row: r, col: c, layer: l + 1 }])
        }
    // Fisher-Yates shuffle
    for (let i = walls.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [walls[i], walls[j]] = [walls[j], walls[i]]
    }
    const count = Math.min(extra, walls.length)
    for (let i = 0; i < count; i++) m.removeWall(walls[i][0], walls[i][1])
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

  btnGen.addEventListener('click', generate)
  btnSolve.addEventListener('click', solve)

  speedInput.addEventListener('input', () => {
    speedLabel.textContent = `${speedInput.value} ms`
  })

  sizeInput.addEventListener('input', () => {
    const v = sizeInput.value
    sizeLabel.textContent  = `${v}×${v}×${v}`
    cubeTitle.textContent  = `Cube 3D — ${v}×${v}×${v}`
  })

  openingsInput.addEventListener('input', () => {
    openingsLabel.textContent = openingsInput.value
  })

  generate()
}

export function resizeCubeCanvas(): void {
  const controls = document.getElementById('cube-controls')!
  const title    = document.getElementById('cube-title')!
  const usedH    = title.offsetHeight + controls.offsetHeight + 100
  const size     = Math.min(window.innerWidth - 40, window.innerHeight - usedH)
  canvas.width   = Math.max(300, size)
  canvas.height  = Math.max(300, size)
  renderer?.resize()
}
