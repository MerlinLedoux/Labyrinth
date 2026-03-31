import type { Maze } from '../core/maze'
import type { Cell, GeneratorStep } from '../core/types'
import type { MazeGenerator } from './generator'

// 1. Grille complette (toute les cellules sont entre 4 murs)
// 2. Une liste avec tous les murs supprimable dans la grille.
// 3. Fonction qui verifie si deux cellules sont lier. (Trouver un chemain a chaque fois c'est trop long donc il faut une autre méthode). Ma solution : Je met toute les case dans une liste de liste tous les ellement de la meme liste sont lier donc ca va trés vite pour vérifier si on supprime ou non le mur.
// 4. On itére la méthode jusqu'a la liste de liste ne contiéne qu'un unique élement.

export class KruskalGenerator implements MazeGenerator {
    async *generate(maze: Maze): AsyncGenerator<GeneratorStep> {
        type Wall = { a: Cell, b: Cell };
        const walls: Wall[] = [];
        const sacs: Cell[][] = [];

        // PERFORMANCE : sameSac() et les findIndex() dans la boucle principale sont en O(n) à chaque appel.
        // Sur une grille N×M on effectue O(N×M) appels, ce qui donne une complexité globale O((N×M)²).
        // Solution : remplacer les "sacs" par un vrai Union-Find avec un tableau parent[] :
        //   - parent[i] = index du représentant du groupe de la cellule i
        //   - find(i) remonte jusqu'à la racine (avec path compression → quasi O(1))
        //   - union(a, b) fusionne les deux groupes en un seul
        // Cela ramènerait la complexité globale à O(N×M × α(N×M)) ≈ O(N×M).
        const sameSac = (a: Cell, b: Cell): boolean => {
            // 1. On cherche l'index du sac qui contient la cellule 'a'
            const indexDuSac = sacs.findIndex(sac => 
                sac.some(cell => cell.row === a.row && cell.col === a.col));

            // Si on n'a pas trouvé de sac pour 'a' (théoriquement impossible ici), on renvoie false
            if (indexDuSac === -1) return false;

            // 2. On regarde si la cellule 'b' est présente dans CE MÊME sac
            const bDansLeSacDeA = sacs[indexDuSac].some(cell => 
                cell.row === b.row && cell.col === b.col);

            return bDansLeSacDeA;
        };

        for (let row = 0; row < maze.rows; row++) {
            for (let col = 0; col < maze.cols; col++) {

                if (row < maze.rows - 1) {
                    walls.push({
                        a: { row: row, col: col },
                        b: { row: row + 1, col: col }
                    });
                }

                if (col < maze.cols - 1) {
                    walls.push({
                        a: { row: row, col: col },
                        b: { row: row, col: col + 1 }
                    });
                }
            }
        }

        // Il n'y a pas de fonction de mélange native dans ts donc on en ajoute une basé sur l'algorithme de Fisher-Yates.
        function shuffle(array: any[]) {
            for (let i = array.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [array[i], array[j]] = [array[j], array[i]];
            }
        }

        shuffle(walls);

        // A l'étape 0 toutes les cellules sont dans des sac différent
        for (let row = 0; row < maze.rows; row++) {
            for (let col = 0; col < maze.cols; col++) {
                sacs.push([{ row: row, col: col }])
            }
        }

        // Début de la génération du labyrinthe
        while (sacs.length > 1 && walls.length > 0) {
 
            // Le ! après pop() dit à TypeScript : "fais-moi confiance, ce ne sera jamais undefined ici". C'est justifié car la condition walls.length > 0 dans le while garantit qu'il y a toujours au moins un élément avant d'appeler pop(). TypeScript ne fait juste pas ce raisonnement tout seul.
            const mur = walls.pop()!;
            // regarder si les deux coté du mure sont dans le même sac
            const cellA = mur.a;
            const cellB = mur.b;
            if (!sameSac(cellA, cellB)) {
                maze.removeWall(cellA, cellB)
                const indexA = sacs.findIndex(sac => sac.some(cell => cell.row === cellA.row && cell.col === cellA.col));
                const indexB = sacs.findIndex(sac => sac.some(cell => cell.row === cellB.row && cell.col === cellB.col));
                sacs[indexA].push(...sacs[indexB]);
                sacs.splice(indexB, 1);
                yield { type: 'visit', cell: cellA }
            }
        }
        yield { type: 'done', cell: { row: 0, col: 0 } }
    }
}