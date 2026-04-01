import { Maze } from './core/maze'
import { NORTH, SOUTH, EAST, WEST } from './core/types'
import type { Cell } from './core/types'

const MAZE_SIZE   = 15
const EXTRA_OPEN  = 20
const VIS_RANGE   = 3

// ── Couleurs ──────────────────────────────────────────────────────────────────
const COL = {
  bg:        '#0d0d0d',
  fog:       '#141414',
  memory:    '#2a2a2a',
  visible:   '#c0c0c0',
  wallVis:   '#000000',
  wallMem:   '#444444',
  player:    '#C94747',
  end:       '#ff8a65',
}

// ── DOM ───────────────────────────────────────────────────────────────────────
const canvas    = document.getElementById('play-canvas')    as HTMLCanvasElement
const ctx       = canvas.getContext('2d')!
const stepEl    = document.getElementById('play-steps')     as HTMLSpanElement
const msgEl     = document.getElementById('play-msg')       as HTMLDivElement
const btnNew    = document.getElementById('btn-play-new')   as HTMLButtonElement

// ── State ─────────────────────────────────────────────────────────────────────
let maze    : Maze | null = null
let player  : Cell  = { row: 0,            col: 0            }
let end     : Cell  = { row: MAZE_SIZE - 1, col: MAZE_SIZE - 1 }
let steps   = 0
let won     = false
let revealed = new Set<number>()   // toutes les cases déjà vues
export let playActive = false      // écoute clavier active seulement ici

// ── BFS visibility ────────────────────────────────────────────────────────────
/** Retourne l'ensemble des cases accessibles en ≤ range pas sans traverser de mur. */
function computeVisible(m: Maze, from: Cell, range: number): Set<number> {
  const vis  = new Set<number>()
  const queue: [Cell, number][] = [[from, 0]]
  vis.add(m.cellKey(from))
  while (queue.length > 0) {
    const [cell, dist] = queue.shift()!
    if (dist >= range) continue
    for (const nb of m.passableNeighbors(cell)) {
      const k = m.cellKey(nb)
      if (!vis.has(k)) {
        vis.add(k)
        queue.push([nb, dist + 1])
      }
    }
  }
  return vis
}

// ── Rendu ─────────────────────────────────────────────────────────────────────
function draw(): void {
  if (!maze) return
  const { rows, cols } = maze
  const cs = Math.floor(Math.min(canvas.width / cols, canvas.height / rows))
  const ox = Math.floor((canvas.width  - cols * cs) / 2)
  const oy = Math.floor((canvas.height - rows * cs) / 2)

  const visible = computeVisible(maze, player, VIS_RANGE)
  // Ajoute toutes les cases visibles au set des cases révélées
  for (const k of visible) revealed.add(k)

  // Fond
  ctx.fillStyle = COL.bg
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const key = maze.cellKey({ row, col })
      const x   = ox + col * cs
      const y   = oy + row * cs
      const isVis = visible.has(key)
      const isMem = !isVis && revealed.has(key)

      if (!isVis && !isMem) {
        // Brouillard : carré sombre, pas de murs
        ctx.fillStyle = COL.fog
        ctx.fillRect(x, y, cs, cs)
        continue
      }

      // Fond cellule
      ctx.fillStyle = isVis ? COL.visible : COL.memory
      ctx.fillRect(x, y, cs, cs)

      // Murs de cette cellule
      ctx.strokeStyle = isVis ? COL.wallVis : COL.wallMem
      ctx.lineWidth   = isVis ? 2 : 1.5
      ctx.beginPath()
      if (maze.hasWall({ row, col }, NORTH)) { ctx.moveTo(x,      y     ); ctx.lineTo(x + cs, y     ) }
      if (maze.hasWall({ row, col }, SOUTH)) { ctx.moveTo(x,      y + cs); ctx.lineTo(x + cs, y + cs) }
      if (maze.hasWall({ row, col }, WEST))  { ctx.moveTo(x,      y     ); ctx.lineTo(x,      y + cs) }
      if (maze.hasWall({ row, col }, EAST))  { ctx.moveTo(x + cs, y     ); ctx.lineTo(x + cs, y + cs) }
      ctx.stroke()
    }
  }

  // Case d'arrivée — toujours visible (même dans le brouillard)
  const ex = ox + end.col * cs
  const ey = oy + end.row * cs
  ctx.fillStyle = COL.end
  ctx.fillRect(ex + 2, ey + 2, cs - 4, cs - 4)
  ctx.fillStyle = '#ffffff'
  ctx.font = `bold ${Math.floor(cs * 0.55)}px sans-serif`
  ctx.textAlign    = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('★', ex + cs / 2, ey + cs / 2)

  // Joueur
  const px = ox + player.col * cs + cs / 2
  const py = oy + player.row * cs + cs / 2
  const pr = cs * 0.32
  ctx.beginPath()
  ctx.arc(px, py, pr, 0, Math.PI * 2)
  ctx.fillStyle = COL.player
  ctx.fill()

  // Message victoire
  if (won) {
    msgEl.textContent = `Bravo ! ${steps} pas`
    msgEl.style.display = 'block'
  }
}

// ── Mouvement ─────────────────────────────────────────────────────────────────
function move(dir: typeof NORTH | typeof SOUTH | typeof EAST | typeof WEST): void {
  if (!maze || won) return

  const deltas: Record<number, [number, number]> = {
    [NORTH]: [-1,  0],
    [SOUTH]: [ 1,  0],
    [EAST]:  [ 0,  1],
    [WEST]:  [ 0, -1],
  }
  const [dr, dc] = deltas[dir]

  if (maze.hasWall(player, dir)) return   // mur : mouvement impossible

  player = { row: player.row + dr, col: player.col + dc }
  steps++
  stepEl.textContent = String(steps)

  if (player.row === end.row && player.col === end.col) {
    won = true
  }

  draw()
}

// ── Clavier ───────────────────────────────────────────────────────────────────
function onKeyDown(e: KeyboardEvent): void {
  if (!playActive) return
  switch (e.key) {
    case 'ArrowUp':    case 'z': case 'Z': e.preventDefault(); move(NORTH); break
    case 'ArrowDown':  case 's': case 'S': e.preventDefault(); move(SOUTH); break
    case 'ArrowRight': case 'd': case 'D': e.preventDefault(); move(EAST);  break
    case 'ArrowLeft':  case 'q': case 'Q': e.preventDefault(); move(WEST);  break
  }
}

// ── Nouvelle partie ───────────────────────────────────────────────────────────
function newGame(): void {
  const m = new Maze(MAZE_SIZE, MAZE_SIZE)

  // Recursive backtracker (DFS) inline — pas de dépendance sur les generators async
  const visited = new Set<number>()
  const stack: Cell[] = [{ row: 0, col: 0 }]
  visited.add(m.cellKey({ row: 0, col: 0 }))
  while (stack.length > 0) {
    const cur  = stack[stack.length - 1]
    const dirs = [NORTH, SOUTH, EAST, WEST]
      .map(d => {
        const dr = d === NORTH ? -1 : d === SOUTH ? 1 : 0
        const dc = d === EAST  ?  1 : d === WEST ? -1 : 0
        return { d, nb: { row: cur.row + dr, col: cur.col + dc } }
      })
      .filter(({ nb }) =>
        m.inBounds(nb.row, nb.col) && !visited.has(m.cellKey(nb))
      )
    if (dirs.length === 0) { stack.pop(); continue }
    const { d: chosenDir, nb } = dirs[Math.floor(Math.random() * dirs.length)]
    m.removeWall(cur, nb)
    visited.add(m.cellKey(nb))
    stack.push(nb)
    void chosenDir  // la direction est déjà appliquée via removeWall
  }

  // Extra openings
  const walls: [Cell, Cell][] = []
  for (let r = 0; r < MAZE_SIZE; r++)
    for (let c = 0; c < MAZE_SIZE; c++) {
      if (c + 1 < MAZE_SIZE && m.hasWall({ row: r, col: c }, EAST))
        walls.push([{ row: r, col: c }, { row: r, col: c + 1 }])
      if (r + 1 < MAZE_SIZE && m.hasWall({ row: r, col: c }, SOUTH))
        walls.push([{ row: r, col: c }, { row: r + 1, col: c }])
    }
  for (let i = walls.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [walls[i], walls[j]] = [walls[j], walls[i]]
  }
  for (let i = 0; i < Math.min(EXTRA_OPEN, walls.length); i++)
    m.removeWall(walls[i][0], walls[i][1])

  maze     = m
  player   = { row: 0, col: 0 }
  end      = { row: MAZE_SIZE - 1, col: MAZE_SIZE - 1 }
  steps    = 0
  won      = false
  revealed = new Set<number>()

  stepEl.textContent  = '0'
  msgEl.style.display = 'none'
  draw()
}

// ── Init & resize ─────────────────────────────────────────────────────────────
export function initPlayApp(): void {
  btnNew.addEventListener('click', newGame)
  window.addEventListener('keydown', onKeyDown)
  newGame()
}

export function resizePlayCanvas(): void {
  const controls = document.getElementById('play-controls')!
  const title    = document.getElementById('play-title')!
  const info     = document.getElementById('play-info')!
  const usedH    = title.offsetHeight + controls.offsetHeight + info.offsetHeight + 80
  const size     = Math.min(window.innerWidth - 40, window.innerHeight - usedH)
  canvas.width   = Math.max(300, size)
  canvas.height  = Math.max(300, size)
  if (maze) draw()
}

export function setPlayActive(active: boolean): void {
  playActive = active
}
