import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { NORTH, SOUTH, EAST, WEST } from '../core/types'
import type { Maze } from '../core/maze'
import type { Cell } from '../core/types'

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  background:  0xC7C7C7,
  wall:        0x736E6E,
  ground:      0x7D7D7D,
  unvisited:   0x7D7D7D,  // herbe — fond quasi-invisible
  inMaze:      0xB8B8B8,  // pierre sable (couloir)
  frontier:    0x90caf9,
  open:        0x9FCF65,
  closed:      0x65ADCF,
  openFaded:   0xAEC196,  // open fondu vers inMaze (chemin trouvé)
  closedFaded: 0x96B3C1,  // closed fondu vers inMaze (chemin trouvé)
  path:        0xC94747,
  start:       0x81d4fa,
  end:         0x9E3C3C,
}

// Hauteurs des dalles selon leur état (géométrie de base h=1, scalée)
const H = {
  unvisited: 0.01,
  inMaze:    0.07,
  frontier:  0.12,
  open:      0.20,
  closed:    0.32,
  path:      0.54,
  marker:    0.40,
}

const WALL_H = 0.7    // hauteur de base des haies
const WALL_T = 0.35   // épaisseur des haies
const WALL_H_VAR = 0.20  // amplitude de variation de hauteur (± 30 %)

// Temporaires réutilisables
const _mat  = new THREE.Matrix4()
const _col  = new THREE.Color()
const _zero = new THREE.Matrix4().makeScale(0, 0, 0)

// ── Textures procédurales ────────────────────────────────────────────────────

/** Texture neutre pour les haies : taches claires/sombres sur fond gris moyen.
 *  La teinte réelle vient de instanceColor × cette texture. */
function makeHedgeTexture(): THREE.CanvasTexture {
  const S = 256
  const cv = document.createElement('canvas')
  cv.width = cv.height = S
  const ctx = cv.getContext('2d')!
  ctx.fillStyle = '#888888'
  ctx.fillRect(0, 0, S, S)
  for (let i = 0; i < 120; i++) {
    const x = Math.random() * S
    const y = Math.random() * S
    const rx = 4 + Math.random() * 10
    const ry = 2 + Math.random() * 6
    const l  = 45 + Math.random() * 30  // lightness 45–75 % (gris)
    ctx.beginPath()
    ctx.ellipse(x, y, rx, ry, Math.random() * Math.PI, 0, Math.PI * 2)
    ctx.fillStyle = `hsl(0,0%,${l}%)`
    ctx.fill()
  }
  const tex = new THREE.CanvasTexture(cv)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  tex.repeat.set(2, 2)
  return tex
}

/** Texture bruit léger : variation subtile pour les dalles colorées. */
function makeNoiseTexture(): THREE.CanvasTexture {
  const S = 128
  const cv = document.createElement('canvas')
  cv.width = cv.height = S
  const ctx = cv.getContext('2d')!
  for (let x = 0; x < S; x += 2) {
    for (let y = 0; y < S; y += 2) {
      const v = 200 + Math.floor(Math.random() * 55)
      ctx.fillStyle = `rgb(${v},${v},${v})`
      ctx.fillRect(x, y, 2, 2)
    }
  }
  return new THREE.CanvasTexture(cv)
}

// ── Classe principale ─────────────────────────────────────────────────────────
export class Renderer3D {
  private scene      = new THREE.Scene()
  private camera     : THREE.PerspectiveCamera
  private glRenderer : THREE.WebGLRenderer
  private controls   : OrbitControls
  private rafId      = 0

  // Géométrie instanciée
  private floorIM!   : THREE.InstancedMesh
  private hWallIM!   : THREE.InstancedMesh
  private vWallIM!   : THREE.InstancedMesh

  // Hauteurs aléatoires par segment de haie (générées une fois par maze)
  private hWallH!    : Float32Array  // (rows+1) * cols
  private vWallH!    : Float32Array  // rows * (cols+1)

  // Marqueurs start/end
  private startMesh  : THREE.Mesh | null = null
  private endMesh    : THREE.Mesh | null = null

  // Plan de sol pour boucher le ciel
  private groundMesh : THREE.Mesh | null = null

  private lastRows = -1
  private lastCols = -1

  // Référence au soleil pour recalibrer son frustum d'ombre à chaque nouvelle grille
  private sun!: THREE.DirectionalLight

  // Textures (créées une fois, partagées)
  private readonly hedgeTex = makeHedgeTexture()
  private readonly noiseTex = makeNoiseTexture()

  constructor(private canvas: HTMLCanvasElement) {
    this.scene.background = new THREE.Color(C.background)

    this.camera = new THREE.PerspectiveCamera(50, canvas.width / canvas.height, 0.1, 500)

    this.glRenderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.glRenderer.setSize(canvas.width, canvas.height, false)
    this.glRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.glRenderer.shadowMap.enabled = true
    this.glRenderer.shadowMap.type    = THREE.PCFSoftShadowMap

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.minDistance   = 2
    this.controls.maxPolarAngle = Math.PI / 2.05

    // Lumière hémisphérique ciel/sol
    this.scene.add(new THREE.HemisphereLight(0x87ceeb, 0x4a7030, 0.75))

    // Soleil — position et frustum recalibrés dans buildGeometry()
    this.sun = new THREE.DirectionalLight(0xfffbe6, 1.0)
    this.sun.castShadow            = true
    this.sun.shadow.mapSize.width  = 2048
    this.sun.shadow.mapSize.height = 2048
    this.sun.shadow.bias           = -0.0008
    this.scene.add(this.sun)
    this.scene.add(this.sun.target)  // la cible doit être dans la scène

    // Lumière de remplissage douce (côté opposé au soleil)
    const fill = new THREE.DirectionalLight(0xc8e0ff, 0.3)
    fill.position.set(-5, 8, -5)
    this.scene.add(fill)

    this.rafId = requestAnimationFrame(this.loop)
  }

  // ─── Boucle de rendu ────────────────────────────────────────────────────────

  private loop = (): void => {
    this.rafId = requestAnimationFrame(this.loop)
    this.controls.update()
    this.glRenderer.render(this.scene, this.camera)
  }

  // ─── API publique ────────────────────────────────────────────────────────────

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
    maze: Maze,
    options: {
      inMaze?   : Set<number>
      frontier? : Set<number>
      open?     : Set<number>
      closed?   : Set<number>
      path?     : Cell[]
      start?    : Cell
      end?      : Cell
    } = {},
  ): void {
    if (maze.rows !== this.lastRows || maze.cols !== this.lastCols) {
      this.buildGeometry(maze)
    }
    this.updateFloorTiles(maze, options)
    this.updateWallVisibility(maze)
    this.updateMarkers(options)
  }

  // ─── Privé ───────────────────────────────────────────────────────────────────

  private buildGeometry(maze: Maze): void {
    const { rows, cols } = maze

    // Nettoyage
    if (this.lastRows !== -1) {
      this.scene.remove(this.floorIM, this.hWallIM, this.vWallIM)
      this.floorIM.geometry.dispose() ; (this.floorIM.material as THREE.Material).dispose()
      this.hWallIM.geometry.dispose() ; (this.hWallIM.material as THREE.Material).dispose()
      this.vWallIM.geometry.dispose() ; (this.vWallIM.material as THREE.Material).dispose()
    }
    if (this.groundMesh) {
      this.scene.remove(this.groundMesh)
      this.groundMesh.geometry.dispose()
      ;(this.groundMesh.material as THREE.Material).dispose()
    }

    // ── Plan de sol — bouche le fond bleu visible à travers les dalles ───────
    this.groundMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(cols + 2, rows + 2),
      new THREE.MeshStandardMaterial({ color: C.ground, roughness: 1, metalness: 0 }),
    )
    this.groundMesh.rotation.x = -Math.PI / 2
    this.groundMesh.position.set((cols - 1) / 2, -0.005, (rows - 1) / 2)
    this.groundMesh.receiveShadow = true
    this.scene.add(this.groundMesh)

    // ── Dalles (géométrie h=1, height contrôlée par scale Y dans la matrice) ─
    //    Deux matériaux : pierre pour inMaze, bruit pour les états colorés
    //    → on utilise un seul material avec noiseTex (instanceColor × texture)
    const nFloor = rows * cols
    this.floorIM = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.96, 1, 0.96),
      new THREE.MeshStandardMaterial({
        map:       this.noiseTex,
        roughness: 0.85,
        metalness: 0,
      }),
      nFloor,
    )
    this.floorIM.castShadow    = true
    this.floorIM.receiveShadow = true
    for (let i = 0; i < nFloor; i++) {
      const h = H.unvisited
      _mat.makeScale(1, h, 1)
      _mat.setPosition(i % cols, h / 2, Math.floor(i / cols))
      this.floorIM.setMatrixAt(i, _mat)
      this.floorIM.setColorAt(i, _col.set(C.unvisited))
    }
    this.floorIM.instanceMatrix.needsUpdate = true
    this.floorIM.instanceColor!.needsUpdate = true
    this.scene.add(this.floorIM)

    // ── Matériau haies ────────────────────────────────────────────────────────
    const wallMat = new THREE.MeshStandardMaterial({
      map:       this.hedgeTex,
      roughness: 0.95,
      metalness: 0,
    })

    // ── Haies horizontales ── hauteur variable aléatoire ──────────────────────
    const nHWall = (rows + 1) * cols
    this.hWallH  = new Float32Array(nHWall)
    this.hWallIM = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1 + WALL_T, 1, WALL_T),  // h=1 (unit), scalée
      wallMat,
      nHWall,
    )
    this.hWallIM.castShadow    = true
    this.hWallIM.receiveShadow = true
    for (let r = 0; r <= rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c
        const h   = WALL_H * (1 - WALL_H_VAR / 2 + Math.random() * WALL_H_VAR)
        this.hWallH[idx] = h
        _mat.makeScale(1, h, 1)
        _mat.setPosition(c, h / 2, r - 0.5)
        this.hWallIM.setMatrixAt(idx, _mat)
        // Variation de teinte par segment
        _col.set(C.wall)
        const hsl = { h: 0, s: 0, l: 0 }
        _col.getHSL(hsl)
        _col.setHSL(hsl.h, hsl.s, hsl.l + (Math.random() - 0.5) * 0.08)
        this.hWallIM.setColorAt(idx, _col)
      }
    }
    this.hWallIM.instanceMatrix.needsUpdate = true
    this.hWallIM.instanceColor!.needsUpdate = true
    this.scene.add(this.hWallIM)

    // ── Haies verticales ── même traitement ────────────────────────────────────
    const nVWall = rows * (cols + 1)
    this.vWallH  = new Float32Array(nVWall)
    this.vWallIM = new THREE.InstancedMesh(
      new THREE.BoxGeometry(WALL_T, 1, 1 + WALL_T),
      wallMat,
      nVWall,
    )
    this.vWallIM.castShadow    = true
    this.vWallIM.receiveShadow = true
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c <= cols; c++) {
        const idx = r * (cols + 1) + c
        const h   = WALL_H * (1 - WALL_H_VAR / 2 + Math.random() * WALL_H_VAR)
        this.vWallH[idx] = h
        _mat.makeScale(1, h, 1)
        _mat.setPosition(c - 0.5, h / 2, r)
        this.vWallIM.setMatrixAt(idx, _mat)
        _col.set(C.wall)
        const hsl = { h: 0, s: 0, l: 0 }
        _col.getHSL(hsl)
        _col.setHSL(hsl.h, hsl.s, hsl.l + (Math.random() - 0.5) * 0.08)
        this.vWallIM.setColorAt(idx, _col)
      }
    }
    this.vWallIM.instanceMatrix.needsUpdate = true
    this.vWallIM.instanceColor!.needsUpdate = true
    this.scene.add(this.vWallIM)

    // ── Caméra ────────────────────────────────────────────────────────────────
    const cx   = (cols - 1) / 2
    const cz   = (rows - 1) / 2
    const dist = Math.max(rows, cols) * 0.9
    this.controls.target.set(cx, 0, cz)
    this.camera.position.set(cx, dist, cz + dist)
    this.camera.lookAt(cx, 0, cz)
    this.controls.update()

    // ── Soleil : repositionné sur le centre + frustum ajusté à la grille ─────
    // Le frustum d'une DirectionalLight est une caméra orthographique.
    // Il doit couvrir toute la scène sinon les ombres disparaissent hors zone.
    const half = Math.max(rows, cols) * 0.65 + 3
    this.sun.position.set(cx + half * 0.6, half * 1.8, cz + half * 0.8)
    this.sun.target.position.set(cx, 0, cz)
    this.sun.target.updateMatrixWorld()

    const sc = this.sun.shadow.camera as THREE.OrthographicCamera
    sc.left   = -half
    sc.right  =  half
    sc.top    =  half
    sc.bottom = -half
    sc.near   = 0.5
    sc.far    = half * 5
    sc.updateProjectionMatrix()

    this.lastRows = rows
    this.lastCols = cols
  }

  /** Couleur + hauteur de chaque dalle selon l'état courant. */
  private updateFloorTiles(maze: Maze, options: {
    inMaze?   : Set<number>
    frontier? : Set<number>
    open?     : Set<number>
    closed?   : Set<number>
    path?     : Cell[]
    start?    : Cell
    end?      : Cell
  }): void {
    const { rows, cols } = maze

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const key = maze.cellKey({ row, col })
        let color = C.unvisited
        let h     = H.unvisited

        const pathFound = (options.path?.length ?? 0) > 0
        if      (options.path?.some(c => c.row === row && c.col === col)) { color = C.path;                              h = H.path     }
        else if (options.closed?.has(key))   { color = pathFound ? C.closedFaded : C.closed; h = H.closed   }
        else if (options.open?.has(key))     { color = pathFound ? C.openFaded   : C.open;   h = H.open     }
        else if (options.frontier?.has(key)) { color = C.frontier;                            h = H.frontier }
        else if (options.inMaze?.has(key))   { color = C.inMaze;                              h = H.inMaze   }

        if (options.start?.row === row && options.start?.col === col) { color = C.start; h = H.marker }
        if (options.end?.row   === row && options.end?.col   === col) { color = C.end;   h = H.marker }

        _mat.makeScale(1, h, 1)
        _mat.setPosition(col, h / 2, row)
        this.floorIM.setMatrixAt(row * cols + col, _mat)
        this.floorIM.setColorAt(row * cols + col, _col.set(color))
      }
    }
    this.floorIM.instanceMatrix.needsUpdate = true
    this.floorIM.instanceColor!.needsUpdate = true
  }

  /** Affiche/cache chaque haie en conservant sa hauteur aléatoire. */
  private updateWallVisibility(maze: Maze): void {
    const { rows, cols } = maze

    for (let r = 0; r <= rows; r++) {
      for (let c = 0; c < cols; c++) {
        let visible: boolean
        if      (r === 0)    visible = maze.hasWall({ row: 0,        col: c }, NORTH)
        else if (r === rows) visible = maze.hasWall({ row: rows - 1, col: c }, SOUTH)
        else                 visible = maze.hasWall({ row: r,        col: c }, NORTH)

        const idx = r * cols + c
        if (visible) {
          const h = this.hWallH[idx]
          _mat.makeScale(1, h, 1)
          _mat.setPosition(c, h / 2, r - 0.5)
        } else {
          _mat.copy(_zero)
        }
        this.hWallIM.setMatrixAt(idx, _mat)
      }
    }
    this.hWallIM.instanceMatrix.needsUpdate = true

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c <= cols; c++) {
        let visible: boolean
        if      (c === 0)    visible = maze.hasWall({ row: r, col: 0        }, WEST)
        else if (c === cols) visible = maze.hasWall({ row: r, col: cols - 1 }, EAST)
        else                 visible = maze.hasWall({ row: r, col: c        }, WEST)

        const idx = r * (cols + 1) + c
        if (visible) {
          const h = this.vWallH[idx]
          _mat.makeScale(1, h, 1)
          _mat.setPosition(c - 0.5, h / 2, r)
        } else {
          _mat.copy(_zero)
        }
        this.vWallIM.setMatrixAt(idx, _mat)
      }
    }
    this.vWallIM.instanceMatrix.needsUpdate = true
  }

  /** Marqueurs cylindriques start/end au-dessus des dalles. */
  private updateMarkers(options: { start?: Cell; end?: Cell }): void {
    if (this.startMesh) this.scene.remove(this.startMesh)
    if (this.endMesh)   this.scene.remove(this.endMesh)
    this.startMesh = null
    this.endMesh   = null

    const geo = new THREE.CylinderGeometry(0.15, 0.18, 0.35, 16)

    if (options.start) {
      this.startMesh = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({ color: C.start, roughness: 0.4, metalness: 0.2 }),
      )
      this.startMesh.position.set(options.start.col, H.marker + 0.18, options.start.row)
      this.startMesh.castShadow = true
      this.scene.add(this.startMesh)
    }
    if (options.end) {
      this.endMesh = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({ color: C.end, roughness: 0.4, metalness: 0.2 }),
      )
      this.endMesh.position.set(options.end.col, H.marker + 0.18, options.end.row)
      this.endMesh.castShadow = true
      this.scene.add(this.endMesh)
    }
  }
}
