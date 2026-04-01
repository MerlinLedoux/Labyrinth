import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { NORTH3, SOUTH3, EAST3, WEST3, UP3, DOWN3 } from '../core/maze3d'
import type { Maze3D, Cell3D } from '../core/maze3d'

// ── Palette (même thème que le renderer 2D/3D) ────────────────────────────────
const C = {
  background: 0x1a1a1a,
  wall:       0xaaaaaa,
  inMaze:     0xcccccc,   // gris clair visible à travers les murs
  frontier:   0x90caf9,
  open:       0x9FCF65,
  closed:     0x65ADCF,
  path:       0xC94747,
  start:      0x81d4fa,
  end:        0xff8a65,
}

const WALL_OPACITY = 0.18   // murs très transparents pour voir l'intérieur
const CELL_OPACITY = 0.55   // cellules semi-transparentes
const WALL_T = 0.06          // épaisseur des cloisons
const CELL_S = 0.72          // cubes un peu plus petits → plus d'air entre eux

const _mat  = new THREE.Matrix4()
const _col  = new THREE.Color()
const _zero = new THREE.Matrix4().makeScale(0, 0, 0)

// ── Renderer ──────────────────────────────────────────────────────────────────
export class RendererCube {
  private scene      = new THREE.Scene()
  private camera     : THREE.PerspectiveCamera
  private glRenderer : THREE.WebGLRenderer
  private controls   : OrbitControls
  private rafId      = 0

  // Instanced meshes
  private cellIM!   : THREE.InstancedMesh
  private nsWallIM! : THREE.InstancedMesh   // cloisons ⊥ axe row  (plan XY, fin en Z)
  private ewWallIM! : THREE.InstancedMesh   // cloisons ⊥ axe col  (plan YZ, fin en X)
  private udWallIM! : THREE.InstancedMesh   // cloisons ⊥ axe layer(plan XZ, fin en Y)

  private lastKey = ''

  constructor(private canvas: HTMLCanvasElement) {
    this.scene.background = new THREE.Color(C.background)

    this.camera = new THREE.PerspectiveCamera(50, canvas.width / canvas.height, 0.1, 500)

    this.glRenderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.glRenderer.setSize(canvas.width, canvas.height, false)
    this.glRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping  = true
    this.controls.dampingFactor  = 0.05
    this.controls.maxPolarAngle  = Math.PI   // rotation libre (vue de dessous possible)

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
    const sun = new THREE.DirectionalLight(0xffffff, 0.6)
    sun.position.set(10, 20, 15)
    this.scene.add(sun)

    this.rafId = requestAnimationFrame(this.loop)
  }

  // ─── Boucle ──────────────────────────────────────────────────────────────────

  private loop = (): void => {
    this.rafId = requestAnimationFrame(this.loop)
    this.controls.update()
    this.glRenderer.render(this.scene, this.camera)
  }

  // ─── API ─────────────────────────────────────────────────────────────────────

  resize(): void {
    this.camera.aspect = this.canvas.width / this.canvas.height
    this.camera.updateProjectionMatrix()
    this.glRenderer.setSize(this.canvas.width, this.canvas.height, false)
  }

  dispose(): void {
    cancelAnimationFrame(this.rafId)
    this.controls.dispose()
    this.glRenderer.dispose()
  }

  drawMaze(
    maze: Maze3D,
    options: {
      inMaze?   : Set<number>
      frontier? : Set<number>
      open?     : Set<number>
      closed?   : Set<number>
      path?     : Cell3D[]
      start?    : Cell3D
      end?      : Cell3D
    } = {},
  ): void {
    const key = `${maze.rows}x${maze.cols}x${maze.layers}`
    if (key !== this.lastKey) {
      this.buildGeometry(maze)
      this.lastKey = key
    }
    this.updateCells(maze, options)
    this.updateWalls(maze)
  }

  // ─── Privé ───────────────────────────────────────────────────────────────────

  private buildGeometry(maze: Maze3D): void {
    const { rows, cols, layers } = maze

    // Nettoyage si changement de taille
    if (this.lastKey) {
      this.scene.remove(this.cellIM, this.nsWallIM, this.ewWallIM, this.udWallIM)
      this.cellIM.geometry.dispose();   (this.cellIM.material as THREE.Material).dispose()
      this.nsWallIM.geometry.dispose()
      this.ewWallIM.geometry.dispose()
      this.udWallIM.geometry.dispose()
      ;(this.nsWallIM.material as THREE.Material).dispose()
    }

    const nCells  = rows * cols * layers
    const nNSWall = (rows + 1) * cols * layers
    const nEWWall = rows * (cols + 1) * layers
    const nUDWall = rows * cols * (layers + 1)

    // ── Cellules (cubes semi-transparents) ────────────────────────────────────
    this.cellIM = new THREE.InstancedMesh(
      new THREE.BoxGeometry(CELL_S, CELL_S, CELL_S),
      new THREE.MeshStandardMaterial({
        transparent: true,
        opacity:     CELL_OPACITY,
        depthWrite:  false,
        roughness:   0.7,
        metalness:   0,
      }),
      nCells,
    )
    for (let i = 0; i < nCells; i++) {
      this.cellIM.setMatrixAt(i, _zero)
      this.cellIM.setColorAt(i, _col.set(C.inMaze))
    }
    this.cellIM.instanceMatrix.needsUpdate = true
    this.cellIM.instanceColor!.needsUpdate = true
    this.cellIM.renderOrder = 1   // dessiné après les murs opaques
    this.scene.add(this.cellIM)

    // ── Matériau murs — très transparent pour voir l'intérieur du cube ────────
    const wallMat = new THREE.MeshStandardMaterial({
      color:       C.wall,
      transparent: true,
      opacity:     WALL_OPACITY,
      depthWrite:  false,
      roughness:   0.9,
      metalness:   0,
    })

    // ── NS walls : fine en Z, couvre XY ──────────────────────────────────────
    // Index: r * cols * layers + c * layers + l
    // Position: (col=c, layer=l, row-boundary=r-0.5) → Three.js (x=c, y=l, z=r-0.5)
    this.nsWallIM = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1.0, 1.0, WALL_T),
      wallMat,
      nNSWall,
    )
    for (let r = 0; r <= rows; r++) {
      for (let c = 0; c < cols; c++) {
        for (let l = 0; l < layers; l++) {
          _mat.makeTranslation(c, l, r - 0.5)
          this.nsWallIM.setMatrixAt(r * cols * layers + c * layers + l, _mat)
        }
      }
    }
    this.nsWallIM.instanceMatrix.needsUpdate = true
    this.scene.add(this.nsWallIM)

    // ── EW walls : fine en X, couvre YZ ──────────────────────────────────────
    // Index: r * (cols+1) * layers + c * layers + l
    // Position: (x=c-0.5, y=l, z=r)
    this.ewWallIM = new THREE.InstancedMesh(
      new THREE.BoxGeometry(WALL_T, 1.0, 1.0),
      wallMat,
      nEWWall,
    )
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c <= cols; c++) {
        for (let l = 0; l < layers; l++) {
          _mat.makeTranslation(c - 0.5, l, r)
          this.ewWallIM.setMatrixAt(r * (cols + 1) * layers + c * layers + l, _mat)
        }
      }
    }
    this.ewWallIM.instanceMatrix.needsUpdate = true
    this.scene.add(this.ewWallIM)

    // ── UD walls : fine en Y, couvre XZ ──────────────────────────────────────
    // Index: r * cols * (layers+1) + c * (layers+1) + l
    // Position: (x=c, y=l-0.5, z=r)
    this.udWallIM = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1.0, WALL_T, 1.0),
      wallMat,
      nUDWall,
    )
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        for (let l = 0; l <= layers; l++) {
          _mat.makeTranslation(c, l - 0.5, r)
          this.udWallIM.setMatrixAt(r * cols * (layers + 1) + c * (layers + 1) + l, _mat)
        }
      }
    }
    this.udWallIM.instanceMatrix.needsUpdate = true
    this.scene.add(this.udWallIM)

    // ── Caméra centrée sur le cube ────────────────────────────────────────────
    const cx   = (cols   - 1) / 2
    const cy   = (layers - 1) / 2
    const cz   = (rows   - 1) / 2
    const dist = Math.max(rows, cols, layers) * 1.3
    this.controls.target.set(cx, cy, cz)
    this.camera.position.set(cx + dist, cy + dist * 0.9, cz + dist * 1.3)
    this.camera.lookAt(cx, cy, cz)
    this.controls.update()
  }

  /** Met à jour la couleur/visibilité de chaque cellule selon son état. */
  private updateCells(
    maze: Maze3D,
    options: {
      inMaze?   : Set<number>
      frontier? : Set<number>
      open?     : Set<number>
      closed?   : Set<number>
      path?     : Cell3D[]
      start?    : Cell3D
      end?      : Cell3D
    },
  ): void {
    const { rows, cols, layers } = maze
    const pathFound = (options.path?.length ?? 0) > 0

    for (let l = 0; l < layers; l++) {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cell: Cell3D = { row: r, col: c, layer: l }
          const key = maze.cellKey(cell)
          const idx = key   // cellKey === index dans le InstancedMesh

          const onPath  = options.path?.some(p => p.row === r && p.col === c && p.layer === l)
          const isStart = options.start?.row === r && options.start?.col === c && options.start?.layer === l
          const isEnd   = options.end?.row   === r && options.end?.col   === c && options.end?.layer   === l

          let color: number | null = null

          if      (onPath)                          color = C.path
          else if (isStart)                         color = C.start
          else if (isEnd)                           color = C.end
          else if (options.closed?.has(key))        color = pathFound ? C.inMaze : C.closed
          else if (options.open?.has(key))          color = pathFound ? C.inMaze : C.open
          else if (options.frontier?.has(key))      color = C.frontier
          else if (options.inMaze?.has(key))        color = C.inMaze
          // unvisited → caché

          if (color !== null) {
            _mat.makeTranslation(c, l, r)
            this.cellIM.setMatrixAt(idx, _mat)
            this.cellIM.setColorAt(idx, _col.set(color))
          } else {
            this.cellIM.setMatrixAt(idx, _zero)
          }
        }
      }
    }
    this.cellIM.instanceMatrix.needsUpdate = true
    this.cellIM.instanceColor!.needsUpdate = true
  }

  /** Affiche/cache chaque cloison selon les murs du labyrinthe. */
  private updateWalls(maze: Maze3D): void {
    const { rows, cols, layers } = maze

    // NS walls
    for (let r = 0; r <= rows; r++) {
      for (let c = 0; c < cols; c++) {
        for (let l = 0; l < layers; l++) {
          const visible =
            r === 0    ? maze.hasWall({ row: 0,      col: c, layer: l }, NORTH3) :
            r === rows ? maze.hasWall({ row: rows-1,  col: c, layer: l }, SOUTH3) :
                         maze.hasWall({ row: r,       col: c, layer: l }, NORTH3)
          const idx = r * cols * layers + c * layers + l
          if (visible) {
            _mat.makeTranslation(c, l, r - 0.5)
            this.nsWallIM.setMatrixAt(idx, _mat)
          } else {
            this.nsWallIM.setMatrixAt(idx, _zero)
          }
        }
      }
    }
    this.nsWallIM.instanceMatrix.needsUpdate = true

    // EW walls
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c <= cols; c++) {
        for (let l = 0; l < layers; l++) {
          const visible =
            c === 0    ? maze.hasWall({ row: r, col: 0,     layer: l }, WEST3) :
            c === cols ? maze.hasWall({ row: r, col: cols-1, layer: l }, EAST3) :
                         maze.hasWall({ row: r, col: c,      layer: l }, WEST3)
          const idx = r * (cols + 1) * layers + c * layers + l
          if (visible) {
            _mat.makeTranslation(c - 0.5, l, r)
            this.ewWallIM.setMatrixAt(idx, _mat)
          } else {
            this.ewWallIM.setMatrixAt(idx, _zero)
          }
        }
      }
    }
    this.ewWallIM.instanceMatrix.needsUpdate = true

    // UD walls
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        for (let l = 0; l <= layers; l++) {
          const visible =
            l === 0      ? maze.hasWall({ row: r, col: c, layer: 0        }, DOWN3) :
            l === layers ? maze.hasWall({ row: r, col: c, layer: layers-1  }, UP3)   :
                           maze.hasWall({ row: r, col: c, layer: l         }, DOWN3)
          const idx = r * cols * (layers + 1) + c * (layers + 1) + l
          if (visible) {
            _mat.makeTranslation(c, l - 0.5, r)
            this.udWallIM.setMatrixAt(idx, _mat)
          } else {
            this.udWallIM.setMatrixAt(idx, _zero)
          }
        }
      }
    }
    this.udWallIM.instanceMatrix.needsUpdate = true
  }
}
