import { Maze } from './core/maze'
import { EAST, SOUTH } from './core/types'
import type { Cell } from './core/types'
import { initCubeApp, resizeCubeCanvas } from './cube-app'
import type { MazeGenerator } from './generators/generator'
import { PrimGenerator } from './generators/prim'
import { RecursiveBacktrackerGenerator } from './generators/recursive-backtracker'
import { KruskalGenerator } from './generators/kruskal'
import { BinaryTreeGenerator } from './generators/binary-tree'
import { HuntAndKillGenerator } from './generators/hunt-and-kill'
import { RecursiveDivisionGenerator } from './generators/recursive-division'
import { AStarSolver } from './solvers/astar'
import { BFSSolver } from './solvers/bfs'
import { DFSSolver } from './solvers/dfs'
import type { MazeSolver } from './solvers/solver'
import { Renderer2D } from './renderer/renderer2d'
import { Renderer3D } from './renderer/renderer3d'

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas2d      = document.getElementById('maze-canvas')    as HTMLCanvasElement
const canvas3d      = document.getElementById('maze-canvas-3d') as HTMLCanvasElement
const btnGen        = document.getElementById('btn-generate')   as HTMLButtonElement
const btnSolve      = document.getElementById('btn-solve')      as HTMLButtonElement
const btnToggle     = document.getElementById('btn-toggle-view') as HTMLButtonElement
const sizeSlider     = document.getElementById('size-slider')      as HTMLInputElement
const sizeLabel      = document.getElementById('size-label')       as HTMLSpanElement
const speedSlider    = document.getElementById('speed-slider')     as HTMLInputElement
const speedLabel     = document.getElementById('speed-label')      as HTMLSpanElement
const openingsSlider = document.getElementById('openings-slider')  as HTMLInputElement
const openingsLabel  = document.getElementById('openings-label')   as HTMLSpanElement
const genSelect     = document.getElementById('gen-select')     as HTMLSelectElement
const solverSelect  = document.getElementById('solver-select')  as HTMLSelectElement

// ── Registries ────────────────────────────────────────────────────────────────
const GENERATORS: Record<string, MazeGenerator> = {
  'binary-tree':           new BinaryTreeGenerator(),
  'hunt-and-kill':         new HuntAndKillGenerator(),
  'prim':                  new PrimGenerator(),
  'recursive-backtracker': new RecursiveBacktrackerGenerator(),
  'recursive-division':    new RecursiveDivisionGenerator(),
  'kruskal':               new KruskalGenerator(),
}

const SOLVERS: Record<string, MazeSolver> = {
  'astar': new AStarSolver(),
  'bfs':   new BFSSolver(),
  'dfs':   new DFSSolver(),
}

// ── State ─────────────────────────────────────────────────────────────────────
let maze    : Maze | null = null
let running               = false
let viewMode: '2d' | '3d' = '2d'

const renderer2d = new Renderer2D(canvas2d)
let   renderer3d : Renderer3D | null = null  // created lazily on first 3D toggle

// Render state — kept in sync so switching view redraws correctly
const rs = {
  inMaze   : new Set<number>(),
  frontier : new Set<number>(),
  open     : new Set<number>(),
  closed   : new Set<number>(),
  path     : [] as Cell[],
  start    : undefined as Cell | undefined,
  end      : undefined as Cell | undefined,
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function getSpeed(): number {
  return Number(speedSlider.value)
}

function resizeCanvas(): void {
  const controls = document.getElementById('controls')!
  const legend   = document.getElementById('legend')!
  const h1       = document.querySelector('h1')!
  const usedH    = h1.offsetHeight + controls.offsetHeight + legend.offsetHeight + 120
  const size     = Math.min(window.innerWidth - 40, window.innerHeight - usedH)
  canvas2d.width  = size
  canvas2d.height = size 
  canvas3d.width  = size
  canvas3d.height = size - 200
}

function setControls(disabled: boolean): void {
  btnGen.disabled       = disabled
  btnSolve.disabled     = disabled
  btnToggle.disabled    = disabled
  genSelect.disabled    = disabled
  solverSelect.disabled = disabled
}

/** Redraw using whichever renderer is currently active. */
function drawCurrent(): void {
  if (!maze) return
  if (viewMode === '2d') {
    renderer2d.drawMaze(maze, rs)
  } else {
    renderer3d!.drawMaze(maze, rs)
  }
}

// ── Generate ──────────────────────────────────────────────────────────────────
async function generate(): Promise<void> {
  if (running) return
  running = true
  setControls(true)

  const gridSize = Number(sizeSlider.value)
  maze = new Maze(gridSize, gridSize)

  rs.inMaze   = new Set()
  rs.frontier = new Set()
  rs.open     = new Set()
  rs.closed   = new Set()
  rs.path     = []
  rs.start    = { row: 0, col: 0 }
  rs.end      = { row: gridSize - 1, col: gridSize - 1 }

  const currentMaze = maze
  const generator = GENERATORS[genSelect.value] ?? GENERATORS['prim']

  for await (const step of generator.generate(currentMaze)) {
    if (step.type === 'visit') {
      rs.inMaze.add(currentMaze.cellKey(step.cell))
      rs.frontier.delete(currentMaze.cellKey(step.cell))
      step.frontier?.forEach(f => rs.frontier.add(currentMaze.cellKey(f)))
      drawCurrent()
    }
    if (step.type === 'done') break
    await delay(getSpeed())
  }

  rs.frontier.clear()

  // ── Suppression aléatoire de murs intérieurs ──────────────────────────────
  const extraOpenings = Number(openingsSlider.value)
  if (extraOpenings > 0) {
    // Collecter tous les murs intérieurs encore présents (paires de cellules adjacentes)
    const walls: [{ row: number; col: number }, { row: number; col: number }][] = []
    for (let r = 0; r < currentMaze.rows; r++) {
      for (let c = 0; c < currentMaze.cols; c++) {
        if (c + 1 < currentMaze.cols && currentMaze.hasWall({ row: r, col: c }, EAST))
          walls.push([{ row: r, col: c }, { row: r, col: c + 1 }])
        if (r + 1 < currentMaze.rows && currentMaze.hasWall({ row: r, col: c }, SOUTH))
          walls.push([{ row: r, col: c }, { row: r + 1, col: c }])
      }
    }
    // Fisher-Yates shuffle puis on prend les N premiers
    for (let i = walls.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [walls[i], walls[j]] = [walls[j], walls[i]]
    }
    const count = Math.min(extraOpenings, walls.length)
    for (let i = 0; i < count; i++) {
      currentMaze.removeWall(walls[i][0], walls[i][1])
    }
  }

  drawCurrent()

  running = false
  setControls(false)
}

// ── Solve ─────────────────────────────────────────────────────────────────────
async function solve(): Promise<void> {
  if (!maze || running) return
  running = true
  setControls(true)

  const currentMaze = maze

  rs.open   = new Set()
  rs.closed = new Set()
  rs.path   = []

  // Fill inMaze with every cell so the full maze appears during solving
  rs.inMaze = new Set()
  for (let r = 0; r < currentMaze.rows; r++)
    for (let c = 0; c < currentMaze.cols; c++)
      rs.inMaze.add(currentMaze.cellKey({ row: r, col: c }))

  const solver = SOLVERS[solverSelect.value] ?? SOLVERS['astar']
  for await (const step of solver.solve(currentMaze, rs.start!, rs.end!)) {
    if (step.type === 'open')  rs.open.add(currentMaze.cellKey(step.cell))
    if (step.type === 'close') { rs.closed.add(currentMaze.cellKey(step.cell)); rs.open.delete(currentMaze.cellKey(step.cell)) }
    if (step.type === 'path')  rs.path = step.path ?? []
    if (step.type === 'done')  break
    drawCurrent()
    await delay(getSpeed())
  }

  drawCurrent()

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

openingsSlider.addEventListener('input', () => {
  openingsLabel.textContent = `${openingsSlider.value}`
})

btnGen.addEventListener('click', generate)
btnSolve.addEventListener('click', solve)

btnToggle.addEventListener('click', () => {
  if (viewMode === '2d') {
    viewMode = '3d'
    canvas2d.style.display = 'none'
    canvas3d.style.display = 'block'
    btnToggle.textContent  = 'Switch to 2D'
    // Create the 3D renderer lazily on first switch
    if (!renderer3d) renderer3d = new Renderer3D(canvas3d)
    else renderer3d.resize()
  } else {
    viewMode = '2d'
    canvas3d.style.display = 'none'
    canvas2d.style.display = 'block'
    btnToggle.textContent  = 'Switch to 3D'
  }
  drawCurrent()
})

// ── Cube 3D section toggle ────────────────────────────────────────────────────
const btnOpenCube  = document.getElementById('btn-open-cube')  as HTMLButtonElement
const btnCubeBack  = document.getElementById('btn-cube-back')  as HTMLButtonElement
const mainSection  = document.getElementById('main-section')   as HTMLDivElement
const cubeSection  = document.getElementById('cube-section')   as HTMLDivElement

let cubeInitialized = false

btnOpenCube.addEventListener('click', () => {
  mainSection.style.display = 'none'
  cubeSection.style.display = 'flex'
  resizeCubeCanvas()
  if (!cubeInitialized) {
    cubeInitialized = true
    initCubeApp()
  }
})

btnCubeBack.addEventListener('click', () => {
  cubeSection.style.display = 'none'
  mainSection.style.display = 'flex'
})

window.addEventListener('resize', () => {
  resizeCanvas()
  renderer3d?.resize()
  drawCurrent()
  if (cubeInitialized) resizeCubeCanvas()
})

// ── Init ──────────────────────────────────────────────────────────────────────
resizeCanvas()
generate()
