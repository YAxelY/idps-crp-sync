# Explication Littérale et Mathématique : `make_mnist.py`

Ce document détaille le fonctionnement interne du script `make_mnist.py`, qui génère le jeu de données "Megapixel MNIST". Il traduit le code Python en concepts mathématiques et logiques clairs.

---

## 1. Objectif Global
Générer de très grandes images ($H \times W$, ex: $1500 \times 1500$ pixels) presque vides, contenant :
1.  **5 chiffres MNIST** (taille $28 \times 28$) dispersés aléatoirement.
2.  **50 motifs de bruit** ("gribouillages") dispersés aléatoirement.
3.  Le tout stocké sous forme **éparse** (sparse) pour économiser la mémoire.

---

## 2. Génération du Bruit (Fonction `_create_noise`)

Le bruit n'est pas aléatoire pixel par pixel (comme un bruit gaussien), mais structurel (lignes droites/gribouillages) pour ressembler à des artefacts d'écriture ou de scan.

### A. Création des Lignes
Le script génère $N_{noise}$ lignes droites dans des patchs de $28 \times 28$.

**Mathématiques de la ligne :**
Une ligne passant par l'origine est définie par l'équation $y = \tan(\theta) \cdot x$.

1.  **Choix de l'angle $\theta$ :**
    L'angle est tiré aléatoirement pour chaque motif $i$ :
    $$ \theta_i \sim \text{Uniforme}(0, \frac{\pi}{2.5}) $$
    (Note : Le code utilise `np.tan` directement sur une distribution uniforme, favorisant certains angles).

2.  **Définition du segment :**
    La longueur maximale sur l'axe $x$ est bornée par la taille du patch ($28$ pixels) et la pente :
    $$ x_{max} = \min(27.49, \frac{27.49}{\tan(\theta_i)}) $$
    On génère ensuite 56 points $x$ équidistants entre $0$ et $x_{max}$.

3.  **Discrétisation (Rasterization) :**
    Les coordonnées continues $(x, y)$ sont arrondies aux entiers les plus proches pour allumer les pixels de la matrice $28 \times 28$ :
    $$ A[i, \text{round}(x), \text{round}(y)] = 1 $$

### B. Complexification du Motif
Pour rendre le bruit moins régulier, on superpose deux lignes et on applique des transformations miroirs.

1.  Soit $A$ la matrice des premières lignes et $B$ une version mélangée de $A$ (lignes avec d'autres angles).
2.  **Miroir (Flip) :** Avec une probabilité $p=0.33$, on inverse l'axe horizontal ou vertical de $B$.
3.  **Superposition :** Le motif final $N$ est l'union logique de $A$ et $B$ :
    $$ \text{Noise}_{binary} = (A + B) > 0 $$

### C. Intensité du bruit
Le bruit n'est pas binaire (0 ou 1), il a une intensité de gris variable.
Pour chaque pixel actif, l'intensité $I$ est tirée aléatoirement :
$$ I \sim \text{Uniforme}(0.8, 1.0) $$
Cela crée des traits gris foncé à noirs.

---

## 3. Placement des Objets (Fonction `_get_positions`)

Le script doit placer 5 chiffres sans qu'ils ne se chevauchent.

**Algorithme de placement (Rejection Sampling) :**
Pour chaque chiffre $k \in \{1..5\}$ d'une image :
1.  Tirer une position candidate $(h, w)$ aléatoirement dans le domaine $[0, H-28] \times [0, W-28]$.
2.  Vérifier la condition de non-chevauchement avec tous les chiffres déjà placés :
    $$ \text{Distance}(P_{new}, P_{exist}) \ge 28 $$
    C'est-à-dire : $|h_{new} - h_{old}| \ge 28$ OU $|w_{new} - w_{old}| \ge 28$.
3.  Si chevauchement : Rejeter et recommencer (étape 1).
4.  Sinon : Accepter et stocker la position.

---

## 4. Construction de l'Image (`MegapixelMNIST.Sample`)

Pour chaque image géante :
1.  On initialise une matrice vide (zéros) de taille $H \times W$.
2.  **Ajout du bruit :** On copie les vignettes de bruit $28 \times 28$ aux positions aléatoires pré-calculées.
    $$ \text{Image}[pos_{noise}] = 255 \times \text{MotifNoise} $$
3.  **Ajout des chiffres :** On copie les chiffres MNIST originaux aux positions valides calculées.
    $$ \text{Image}[pos_{digit}] = 255 \times \text{DigitMNIST} $$

> **Note importante :** Les chiffres sont ajoutés *après* le bruit. Si un bruit et un chiffre se chevauchaient (ce qui est possible car on ne vérifie pas le chevauchement bruit-chiffre, seulement chiffre-chiffre), le chiffre écraserait le bruit.

---

## 5. Sparsification (`sparsify`)

Une image de $1500 \times 1500$ floats pèse environ 9 Mo. 5000 images pèseraient 45 Go.
Comme l'image est majoritairement vide (noire = 0), on ne stocke que les pixels non nuls.

**Encodage Sparse (COO - Coordinate List compressée) :**
La fonction `to_sparse(x)` transforme l'image $x$ aplatie (1D) :
1.  Trouver les indices $k$ où $x[k] \neq 0$.
2.  Stocker uniquement le tuple `(indices, valeurs)`.

$$ \text{Data} \approx \{ (k, x[k]) \mid x[k] > 0 \} $$

Ceci réduit drastiquement la taille du fichier final `.npy`.

---

## 6. Étiquetage Multi-Tâches (`_get_numbers`)

Le script génère plusieurs types de labels pour tester différentes capacités du modèle :

1.  **Label cible ($y$) :** On choisit un chiffre cible (ex: "3") uniformément.
    - L'image contient obligatoirement **3 instances** de ce chiffre cible.
    - Elle contient **2 instances** d'autres chiffres (distracteurs).
    - *Objectif :* Classification binaire ("Est-ce que cette image est de classe 3 ?").

2.  **Label Max ($y_{max}$) :** Le plus grand chiffre présent dans l'image.
    $$ y_{max} = \max(\{d_1, d_2, d_3, d_4, d_5\}) $$

3.  **Label Top ($y_{top}$) :** Le chiffre situé le plus haut physiquement dans l'image (plus petite coordonnée $h$).
    $$ k_{top} = \arg\min_k (h_k) \implies y_{top} = d_{k_{top}} $$
    *Pré-requis :* Comprendre que l'axe vertical des images informatiques commence à 0 en haut et augmente vers le bas.

4.  **Label Multi-label ($y_{multi}$) :** Vecteur One-Hot indiquant la présence/absence de chaque chiffre (0-9).
