# Explication Littérale et Mathématique : `mnist_dataset.py`

Ce document explique le fonctionnement du script `mnist_dataset.py` qui charge les données générées et les prépare pour l'entraînement du modèle IDPS-CRP. Il détaille notamment la reconstruction à partir du format sparse et la logique des deux passes (Scout et Learner).

---

## 1. Reconstruction de l'Image (`__getitem__`)

Le Dataset stocke les données sous format compressé (indices, valeurs). À chaque appel pour charger une image $i$, il doit la reconstruire.

### A. Décompression Sparse
L'opération inverse de la sparsification est effectuée :
1.  Allocation d'un vecteur vide (zéros) de taille $H \times W$ (ex: $1500 \times 1500 = 2,250,000$ pixels).
2.  Remplissage aux indices spécifiques :
    $$ \text{Image}_{\text{flat}}[\text{indices}] = \text{valeurs} $$
3.  Remise en forme (Reshape) en matrice 2D $(H, W, 1)$.

### B. Transformation en Patchs (Mosaïque)
Le modèle IDPS ne traite pas l'image entière d'un bloc, mais une grille de petits carrés ("patchs").

**Opération `unfold` (Dépliement) :**
L'image $(1, H, W)$ est découpée en fenêtres glissantes de taille $P \times P$ (ex: $50 \times 50$) avec un pas $S$ (ex: $50$).

-   Si $H=1500$ et $P=50$, on obtient $1500/50 = 30$ patchs en hauteur.
-   Idem en largeur $\implies 30 \times 30 = 900$ patchs au total.

Le tenseur final a la forme :
$$ (N_{patchs}, C, P, P) \approx (900, 1, 50, 50) $$

C'est ce tenseur, appelé "bag" (sac de patchs), qui est envoyé au modèle. L'image originale géante a disparu.

---

## 2. Logique "Two-Pass" (Deux Passes)

C'est le cœur de l'optimisation IDPS. Le Dataset fournit deux méthodes distinctes pour nourrir les deux étapes du modèle.

### A. Passe 1 : Le Scout (`get_scout_data`)

**Objectif :** Fournir une vue globale rapide pour que le modèle "Scout" puisse estimer quels patchs sont intéressants.

**Mécanisme (Downsampling optionnel) :**
Pour aller encore plus vite, on peut réduire la résolution des patchs.
Si l'option `downsample` est activée :
1.  On prend tous les $N$ patchs de résolution $P \times P$ (ex: $50 \times 50$).
2.  On applique une interpolation bilinéaire pour les réduire à $p \times p$ (ex: $12 \times 12$).
    $$ I_{small} = \text{Interpolate}(I_{big}, \text{scale}=0.24) $$
3.  Le tenseur de sortie est très léger : $(B, N, 1, 12, 12)$.

Cela permet au "Scout" de voir *toute* l'image (les 900 patchs) mais en basse résolution, consommant très peu de mémoire.

---

## 3. Passe 2 : Le Learner (`get_learner_data`)

**Objectif :** Fournir les données haute résolution uniquement pour les patchs sélectionnés par le Scout.

**Entrée :**
-   Le batch complet de données (les 900 patchs HD sont en mémoire système ou GPU).
-   Les indices $idx$ sélectionnés par le réseau Scout (taille $M$, ex: les 10 meilleurs patchs).

**Opération de Sélection (Gathering) :**
On ne veut pas copier les 900 patchs dans le réseau de classification (trop lourd). On extrait uniquement les $M$ élus.

Mathématiquement, pour chaque image $b$ du batch :
$$ \text{Patchs}_{finaux}[b] = \{ \text{Patchs}_{initiaux}[b, k] \mid k \in \text{indices}[b] \} $$

**Implémentation Tensorielle :**
Pour faire cela efficacement en parallèle sur GPU sans boucle `for` lente :
1.  On crée un indice de batch étendu pour correspondre à la forme des indices de patchs.
2.  On utilise l'indexation avancée ("Fancy Indexing") :
    ```python
    selected = input[batch_idx, indices]
    ```

**Résultat :** Un tenseur de taille $(B, M, 1, 50, 50)$ qui est beaucoup plus petit que le sac original $(B, 900, 1, 50, 50)$. C'est ce qui permet d'entraîner sur des images géantes sans exploser la VRAM.

---

## 4. Résumé du Flux de Données

1.  **Disque :** Fichier `.npy` compressé (indices, valeurs).
2.  **Chargement (`__getitem__`) :** Décompression $\to$ Image 1500x1500px $\to$ Découpage en 900 patchs $50 \times 50$.
3.  **Modèle Passe 1 :** Appel de `get_scout_data`. Redimensionne (optionnel) vers $12 \times 12$. Le modèle "voit" tout en flou et choisit 10 indices.
4.  **Modèle Passe 2 :** Appel de `get_learner_data`. Utilise les 10 indices pour extraire les vrais patchs $50 \times 50$ HD. Le modèle classifie précisément ces 10 patchs.
