/**
 * Inférence du modèle DQN dans le browser — zéro dépendance.
 *
 * Le fichier public/model_weights.json est généré par python/export_model.py.
 * Architecture : MLP avec couches [232 → 128 → 128 → 4], activation ReLU.
 *
 * Vecteur d'observation (232 valeurs) — IDENTIQUE à maze_env.py :
 *   [0:225]   mémoire — bitmask murs de chaque cellule (N=1,E=2,S=4,W=8),
 *              -1 si la cellule n'a pas encore été observée.
 *   [225]     position joueur — indice normalisé ∈ [0, 1]
 *   [226:230] vision immédiate — [N, E, S, W] : 1=mur, 0=libre
 *   [230:232] delta vers objectif — [dy, dx] normalisés ∈ [−1, 1]
 */
import type { Maze } from './core/maze'
import { NORTH, SOUTH, EAST, WEST } from './core/types'
import type { Cell } from './core/types'

const VIS_RANGE = 3
const RAYS: [typeof NORTH | typeof SOUTH | typeof EAST | typeof WEST, number, number][] = [
  [NORTH, -1,  0],
  [SOUTH,  1,  0],
  [EAST,   0,  1],
  [WEST,   0, -1],
]

// Bitmask observation : N=1, E=2, S=4, W=8 (convention utilisateur)
const OBS_N = 1, OBS_E = 2, OBS_S = 4, OBS_W = 8

// ── Types ─────────────────────────────────────────────────────────────────────
interface Layer { w: number[][]; b: number[] }
interface ModelWeights {
  obs_dim:    number
  action_dim: number
  activation: 'relu' | 'tanh'
  layers:     Layer[]
}

// Actions : index → direction
const ACTION_DIRS = [NORTH, SOUTH, EAST, WEST] as const

// ── Inférence ─────────────────────────────────────────────────────────────────

/** Multiplication matrice-vecteur + biais : y = W·x + b */
function linear(layer: Layer, x: number[]): number[] {
  return layer.w.map((row, i) =>
    row.reduce((sum, w, j) => sum + w * x[j], 0) + layer.b[i]
  )
}

const ACTIVATIONS = {
  relu: (x: number) => Math.max(0, x),
  tanh: Math.tanh,
}

/** Forward pass complet — retourne l'index de l'action (0=N,1=S,2=E,3=W). */
function forward(weights: ModelWeights, obs: number[]): number {
  const activate = ACTIVATIONS[weights.activation]
  let x = obs
  for (const layer of weights.layers.slice(0, -1)) {
    x = linear(layer, x).map(activate)
  }
  const logits = linear(weights.layers[weights.layers.length - 1], x)
  return logits.indexOf(Math.max(...logits))
}

// ── Bitmask murs d'une cellule (convention observation) ───────────────────────
function cellObsBits(maze: Maze, cell: Cell): number {
  return (
    (maze.hasWall(cell, NORTH) ? OBS_N : 0) |
    (maze.hasWall(cell, EAST)  ? OBS_E : 0) |
    (maze.hasWall(cell, SOUTH) ? OBS_S : 0) |
    (maze.hasWall(cell, WEST)  ? OBS_W : 0)
  )
}

// ── Classe agent ──────────────────────────────────────────────────────────────

export class AIAgent {
  private weights : ModelWeights | null = null
  private loaded   = false

  /** Mémoire persistante : une valeur par cellule (-1 = jamais observée). */
  private memory   = new Float32Array(0)
  private memSize  = 0

  async load(url = '/model_weights.json'): Promise<boolean> {
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      this.weights = await res.json() as ModelWeights
      this.loaded  = true
      console.log('[AIAgent] Modèle chargé —', this.weights.layers.length, 'couches')
      return true
    } catch (e) {
      console.warn('[AIAgent] Modèle non disponible :', e)
      return false
    }
  }

  get isLoaded(): boolean { return this.loaded }

  /** Réinitialise la mémoire — à appeler au début de chaque nouvelle partie. */
  reset(maze: Maze): void {
    this.memSize = maze.rows * maze.cols
    this.memory  = new Float32Array(this.memSize).fill(-1)
  }

  /** Retourne la direction choisie par le modèle. */
  pickAction(maze: Maze, player: Cell, end: Cell): typeof NORTH | typeof SOUTH | typeof EAST | typeof WEST {
    if (!this.weights) throw new Error('Modèle non chargé')

    // 1. Mettre à jour la mémoire avec les cellules actuellement visibles
    this._updateMemory(maze, player)

    // 2. Construire l'observation et faire une inférence
    const obs    = this._buildObs(maze, player, end)
    const action = forward(this.weights, obs)
    return ACTION_DIRS[action]
  }

  // ── Privé ──────────────────────────────────────────────────────────────────

  private _updateMemory(maze: Maze, player: Cell): void {
    // Si reset() n'a pas encore été appelé (sécurité)
    if (this.memSize === 0) this.reset(maze)

    this._memorize(maze, player)

    for (const [dir, dr, dc] of RAYS) {
      let cur = { ...player }
      for (let i = 0; i < VIS_RANGE; i++) {
        if (maze.hasWall(cur, dir)) break
        cur = { row: cur.row + dr, col: cur.col + dc }
        this._memorize(maze, cur)
      }
    }
  }

  private _memorize(maze: Maze, cell: Cell): void {
    const idx = cell.row * maze.cols + cell.col
    if (idx >= 0 && idx < this.memSize) {
      this.memory[idx] = cellObsBits(maze, cell)
    }
  }

  private _buildObs(maze: Maze, player: Cell, end: Cell): number[] {
    const size    = maze.rows
    const total   = size * size
    const r = player.row, c = player.col

    // Bloc 1 : mémoire (225 valeurs)
    const mem = Array.from(this.memory)

    // Bloc 2 : position joueur normalisée (1 valeur)
    const playerIdx = (r * maze.cols + c) / (total - 1)

    // Bloc 3 : vision immédiate (4 valeurs, N,E,S,W)
    const immediate = [
      maze.hasWall(player, NORTH) ? 1 : 0,
      maze.hasWall(player, EAST)  ? 1 : 0,
      maze.hasWall(player, SOUTH) ? 1 : 0,
      maze.hasWall(player, WEST)  ? 1 : 0,
    ]

    // Bloc 4 : delta vers objectif normalisé (2 valeurs)
    const goalDelta = [
      (end.row - r) / (size - 1),
      (end.col - c) / (size - 1),
    ]

    return [...mem, playerIdx, ...immediate, ...goalDelta]
  }
}
