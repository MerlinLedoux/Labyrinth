import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { SOUTH3, EAST3, UP3 } from '../core/maze3d'
import type { Maze3D, Cell3D } from '../core/maze3d'

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  background: 0x1a1a1a,
  sphere:     0x3a3a3a,   // nœud non visité
  inMaze:     0x777777,   // nœud/tube dans le labyrinthe généré
  frontier:   0x90caf9,
  open:       0x9FCF65,
  closed:     0x65ADCF,
  path:       0xC94747,
  start:      0x81d4fa,
  end:        0xff8a65,
  dim:        0x1e1e1e,   // tout ce qui n'est pas le chemin une fois trouvé
}

// Priorité visuelle pour choisir la couleur d'un tube entre deux nœuds
const PRIORITY: Record<number, number> = {
  [C.path]:     6,
  [C.start]:    5,
  [C.end]:      5,
  [C.closed]:   4,
  [C.open]:     3,
  [C.frontier]: 2,
  [C.inMaze]:   1,
  [C.sphere]:   0,
  [C.dim]:      -1,
}

const SPHERE_R = 0.10
const TUBE_R   = 0.065

// Rotations pré-calculées pour orienter les cylindres (Three.js : cylindre le long de Y par défaut)
const ROT_NS = new THREE.Matrix4().makeRotationX(Math.PI / 2)   // Y → Z (axe row)
const ROT_EW = new THREE.Matrix4().makeRotationZ(-Math.PI / 2)  // Y → X (axe col)
// UP direction : pas de rotation (Y = axe layer)

const _m   = new THREE.Matrix4()
const _col = new THREE.Color()
const _zero = new THREE.Matrix4().makeScale(0, 0, 0)

type DrawOptions = {
  inMaze?   : Set<number>
  frontier? : Set<number>
  open?     : Set<number>
  closed?   : Set<number>
  path?     : Cell3D[]
  start?    : Cell3D
  end?      : Cell3D
}

// ── Renderer ──────────────────────────────────────────────────────────────────
export class RendererCube {
  private scene      = new THREE.Scene()
  private camera     : THREE.PerspectiveCamera
  private glRenderer : THREE.WebGLRenderer
  private controls   : OrbitControls
  private rafId      = 0

  private sphereIM! : THREE.InstancedMesh
  private nsTubeIM! : THREE.InstancedMesh   // passages SOUTH (axe row)
  private ewTubeIM! : THREE.InstancedMesh   // passages EAST  (axe col)
  private udTubeIM! : THREE.InstancedMesh   // passages UP    (axe layer)

  private lastKey = ''

  constructor(private canvas: HTMLCanvasElement) {
    this.scene.background = new THREE.Color(C.background)
    this.camera = new THREE.PerspectiveCamera(50, canvas.width / canvas.height, 0.1, 500)

    this.glRenderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.glRenderer.setSize(canvas.width, canvas.height, false)
    this.glRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.maxPolarAngle = Math.PI

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9))
    const sun = new THREE.DirectionalLight(0xffffff, 0.5)
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

  drawMaze(maze: Maze3D, options: DrawOptions = {}): void {
    const key = `${maze.rows}x${maze.cols}x${maze.layers}`
    if (key !== this.lastKey) {
      this.buildGeometry(maze)
      this.lastKey = key
    }
    this.updateSpheres(maze, options)
    this.updateTubes(maze, options)
  }

  // ─── Privé ───────────────────────────────────────────────────────────────────

  private buildGeometry(maze: Maze3D): void {
    const { rows, cols, layers } = maze

    // Nettoyage
    if (this.lastKey) {
      this.scene.remove(this.sphereIM, this.nsTubeIM, this.ewTubeIM, this.udTubeIM)
      for (const im of [this.sphereIM, this.nsTubeIM, this.ewTubeIM, this.udTubeIM]) {
        im.geometry.dispose()
        ;(im.material as THREE.Material).dispose()
      }
    }

    const nSpheres = rows * cols * layers
    const nNSTube  = (rows - 1) * cols * layers
    const nEWTube  = rows * (cols - 1) * layers
    const nUDTube  = rows * cols * (layers - 1)

    // ── Sphères ───────────────────────────────────────────────────────────────
    this.sphereIM = new THREE.InstancedMesh(
      new THREE.SphereGeometry(SPHERE_R, 8, 6),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.1 }),
      nSpheres,
    )
    for (let l = 0; l < layers; l++)
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const idx = maze.cellKey({ row: r, col: c, layer: l })
          _m.makeTranslation(c, l, r)
          this.sphereIM.setMatrixAt(idx, _m)
          this.sphereIM.setColorAt(idx, _col.set(C.sphere))
        }
    this.sphereIM.instanceMatrix.needsUpdate = true
    this.sphereIM.instanceColor!.needsUpdate = true
    this.scene.add(this.sphereIM)

    // Matériau partagé pour les tubes (blanc : la couleur vient de instanceColor)
    const tubeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.15 })
    const tubeGeo = new THREE.CylinderGeometry(TUBE_R, TUBE_R, 1.0, 8)

    // ── NS tubes (SOUTH — le long de Z) ──────────────────────────────────────
    this.nsTubeIM = new THREE.InstancedMesh(tubeGeo, tubeMat, nNSTube)
    for (let i = 0; i < nNSTube; i++) {
      this.nsTubeIM.setMatrixAt(i, _zero)
      this.nsTubeIM.setColorAt(i, _col.set(C.inMaze))
    }
    this.nsTubeIM.instanceMatrix.needsUpdate = true
    this.nsTubeIM.instanceColor!.needsUpdate = true
    this.scene.add(this.nsTubeIM)

    // ── EW tubes (EAST — le long de X) ───────────────────────────────────────
    this.ewTubeIM = new THREE.InstancedMesh(tubeGeo, tubeMat, nEWTube)
    for (let i = 0; i < nEWTube; i++) {
      this.ewTubeIM.setMatrixAt(i, _zero)
      this.ewTubeIM.setColorAt(i, _col.set(C.inMaze))
    }
    this.ewTubeIM.instanceMatrix.needsUpdate = true
    this.ewTubeIM.instanceColor!.needsUpdate = true
    this.scene.add(this.ewTubeIM)

    // ── UD tubes (UP — le long de Y) ─────────────────────────────────────────
    this.udTubeIM = new THREE.InstancedMesh(tubeGeo, tubeMat, nUDTube)
    for (let i = 0; i < nUDTube; i++) {
      this.udTubeIM.setMatrixAt(i, _zero)
      this.udTubeIM.setColorAt(i, _col.set(C.inMaze))
    }
    this.udTubeIM.instanceMatrix.needsUpdate = true
    this.udTubeIM.instanceColor!.needsUpdate = true
    this.scene.add(this.udTubeIM)

    // ── Caméra ────────────────────────────────────────────────────────────────
    const cx   = (cols   - 1) / 2
    const cy   = (layers - 1) / 2
    const cz   = (rows   - 1) / 2
    const dist = Math.max(rows, cols, layers) * 1.4
    this.controls.target.set(cx, cy, cz)
    this.camera.position.set(cx + dist, cy + dist, cz + dist * 1.1)
    this.camera.lookAt(cx, cy, cz)
    this.controls.update()
  }

  /** Couleur d'un nœud selon son état courant. */
  private nodeColor(
    key: number,
    options: DrawOptions,
    pathKeys: Set<number>,
    startKey: number,
    endKey: number,
    pathFound: boolean,
  ): number {
    if (key === startKey) return C.start
    if (key === endKey)   return C.end
    if (pathFound) return pathKeys.has(key) ? C.path : C.dim
    if (pathKeys.has(key))          return C.path
    if (options.closed?.has(key))   return C.closed
    if (options.open?.has(key))     return C.open
    if (options.frontier?.has(key)) return C.frontier
    if (options.inMaze?.has(key))   return C.inMaze
    return C.sphere
  }

  private updateSpheres(maze: Maze3D, options: DrawOptions): void {
    const { rows, cols, layers } = maze
    const pathKeys  = new Set(options.path?.map(p => maze.cellKey(p)) ?? [])
    const startKey  = options.start ? maze.cellKey(options.start) : -1
    const endKey    = options.end   ? maze.cellKey(options.end)   : -1
    const pathFound = pathKeys.size > 0

    for (let l = 0; l < layers; l++)
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          const key = maze.cellKey({ row: r, col: c, layer: l })
          this.sphereIM.setColorAt(key, _col.set(
            this.nodeColor(key, options, pathKeys, startKey, endKey, pathFound),
          ))
        }
    this.sphereIM.instanceColor!.needsUpdate = true
  }

  private updateTubes(maze: Maze3D, options: DrawOptions): void {
    const { rows, cols, layers } = maze
    const pathKeys  = new Set(options.path?.map(p => maze.cellKey(p)) ?? [])
    const startKey  = options.start ? maze.cellKey(options.start) : -1
    const endKey    = options.end   ? maze.cellKey(options.end)   : -1
    const pathFound = pathKeys.size > 0

    // Ensemble des arêtes du chemin : "min-max" pour lookup O(1)
    const pathEdges = new Set<number>()
    const path = options.path ?? []
    for (let i = 0; i < path.length - 1; i++) {
      const a = maze.cellKey(path[i])
      const b = maze.cellKey(path[i + 1])
      pathEdges.add(a * rows * cols * layers + b)
      pathEdges.add(b * rows * cols * layers + a)
    }

    const nc = (keyA: number, keyB: number): number => {
      // Tube chemin
      if (pathEdges.has(keyA * rows * cols * layers + keyB)) return C.path
      if (pathFound) return C.dim
      // Sinon : couleur de l'extrémité de plus haute priorité
      const ca = this.nodeColor(keyA, options, pathKeys, startKey, endKey, false)
      const cb = this.nodeColor(keyB, options, pathKeys, startKey, endKey, false)
      return (PRIORITY[ca] ?? 0) >= (PRIORITY[cb] ?? 0) ? ca : cb
    }

    // ── NS tubes ─────────────────────────────────────────────────────────────
    for (let r = 0; r < rows - 1; r++)
      for (let c = 0; c < cols; c++)
        for (let l = 0; l < layers; l++) {
          const idx  = r * cols * layers + c * layers + l
          const open = !maze.hasWall({ row: r, col: c, layer: l }, SOUTH3)
          if (open) {
            _m.copy(ROT_NS); _m.setPosition(c, l, r + 0.5)
            const kA = maze.cellKey({ row: r,     col: c, layer: l })
            const kB = maze.cellKey({ row: r + 1, col: c, layer: l })
            this.nsTubeIM.setMatrixAt(idx, _m)
            this.nsTubeIM.setColorAt(idx, _col.set(nc(kA, kB)))
          } else {
            this.nsTubeIM.setMatrixAt(idx, _zero)
          }
        }
    this.nsTubeIM.instanceMatrix.needsUpdate = true
    this.nsTubeIM.instanceColor!.needsUpdate = true

    // ── EW tubes ─────────────────────────────────────────────────────────────
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols - 1; c++)
        for (let l = 0; l < layers; l++) {
          const idx  = r * (cols - 1) * layers + c * layers + l
          const open = !maze.hasWall({ row: r, col: c, layer: l }, EAST3)
          if (open) {
            _m.copy(ROT_EW); _m.setPosition(c + 0.5, l, r)
            const kA = maze.cellKey({ row: r, col: c,     layer: l })
            const kB = maze.cellKey({ row: r, col: c + 1, layer: l })
            this.ewTubeIM.setMatrixAt(idx, _m)
            this.ewTubeIM.setColorAt(idx, _col.set(nc(kA, kB)))
          } else {
            this.ewTubeIM.setMatrixAt(idx, _zero)
          }
        }
    this.ewTubeIM.instanceMatrix.needsUpdate = true
    this.ewTubeIM.instanceColor!.needsUpdate = true

    // ── UD tubes ─────────────────────────────────────────────────────────────
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        for (let l = 0; l < layers - 1; l++) {
          const idx  = r * cols * (layers - 1) + c * (layers - 1) + l
          const open = !maze.hasWall({ row: r, col: c, layer: l }, UP3)
          if (open) {
            _m.makeTranslation(c, l + 0.5, r)   // cylindre déjà le long de Y
            const kA = maze.cellKey({ row: r, col: c, layer: l     })
            const kB = maze.cellKey({ row: r, col: c, layer: l + 1 })
            this.udTubeIM.setMatrixAt(idx, _m)
            this.udTubeIM.setColorAt(idx, _col.set(nc(kA, kB)))
          } else {
            this.udTubeIM.setMatrixAt(idx, _zero)
          }
        }
    this.udTubeIM.instanceMatrix.needsUpdate = true
    this.udTubeIM.instanceColor!.needsUpdate = true
  }
}
